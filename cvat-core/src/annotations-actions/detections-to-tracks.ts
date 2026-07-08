// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import ObjectState from '../object-state';
import { Job, Task } from '../session';
import { SerializedShape, SerializedTrack, SerializedAttributes } from '../server-response-types';
import { ObjectType, ShapeType, Source } from '../enums';

import { ActionParameters } from './base-action';
import { BaseCollectionAction, CollectionActionInput, CollectionActionOutput } from './base-collection-action';

type Box = [number, number, number, number]; // xtl, ytl, xbr, ybr

// ByteTrack-style association tuning
const MATCH_IOU = 0.3; // min IoU to link a detection to an existing track
const MAX_AGE = 30; // frames a track may coast unmatched before it is closed
const MIN_HITS = 1; // minimum observations for a track to be kept

function iou(a: Box, b: Box): number {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
    const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
    return inter / (areaA + areaB - inter + 1e-9);
}

interface InternalTrack {
    box: Box;
    velocity: Box;
    lastObsBox: Box;
    lastObsFrame: number;
    timeSinceUpdate: number;
    labelID: number;
    observations: Map<number, SerializedShape>; // frame -> original detection shape
}

export class DetectionsToTracks extends BaseCollectionAction {
    #mutableSpecIDs: Set<number> = new Set<number>();

    public async init(instance: Job | Task): Promise<void> {
        this.#mutableSpecIDs = new Set<number>();
        for (const label of instance.labels) {
            for (const attr of label.attributes) {
                if (attr.mutable && typeof attr.id === 'number') {
                    this.#mutableSpecIDs.add(attr.id);
                }
            }
        }
    }

    public async destroy(): Promise<void> {
        // nothing to destroy
    }

    private splitAttributes(
        attrs: SerializedAttributes,
    ): { immutable: SerializedAttributes; mutable: SerializedAttributes } {
        const immutable: SerializedAttributes = [];
        const mutable: SerializedAttributes = [];
        for (const attr of attrs) {
            (this.#mutableSpecIDs.has(attr.spec_id) ? mutable : immutable).push(attr);
        }
        return { immutable, mutable };
    }

    public async run(input: CollectionActionInput): Promise<CollectionActionOutput> {
        const { collection, onProgress } = input;

        const rectangles = collection.shapes.filter(
            (shape) => shape.type === ShapeType.RECTANGLE && Array.isArray(shape.points),
        );

        if (!rectangles.length) {
            return {
                created: { shapes: [], tags: [], tracks: [] },
                deleted: { shapes: [], tags: [], tracks: [] },
            };
        }

        const maxFrame = rectangles.reduce((acc, shape) => Math.max(acc, shape.frame), 0);

        // group detections per label; associate within each label independently
        const byLabel = new Map<number, SerializedShape[]>();
        for (const shape of rectangles) {
            const list = byLabel.get(shape.label_id) ?? [];
            list.push(shape);
            byLabel.set(shape.label_id, list);
        }

        const finishedTracks: InternalTrack[] = [];
        let processedLabels = 0;

        for (const [labelID, shapes] of byLabel.entries()) {
            const perFrame = new Map<number, SerializedShape[]>();
            for (const shape of shapes) {
                const list = perFrame.get(shape.frame) ?? [];
                list.push(shape);
                perFrame.set(shape.frame, list);
            }
            const labelFrames = Array.from(perFrame.keys()).sort((a, b) => a - b);
            let active: InternalTrack[] = [];

            for (const frame of labelFrames) {
                const dets = perFrame.get(frame) as SerializedShape[];
                const detBoxes: Box[] = dets.map((d) => [
                    (d.points as number[])[0], (d.points as number[])[1],
                    (d.points as number[])[2], (d.points as number[])[3],
                ]);

                // constant-velocity prediction
                for (const trk of active) {
                    trk.box = [
                        trk.box[0] + trk.velocity[0], trk.box[1] + trk.velocity[1],
                        trk.box[2] + trk.velocity[2], trk.box[3] + trk.velocity[3],
                    ];
                    trk.timeSinceUpdate += 1;
                }

                // greedy IoU matching
                const pairs: [number, number, number][] = [];
                for (let t = 0; t < active.length; t++) {
                    for (let d = 0; d < detBoxes.length; d++) {
                        pairs.push([iou(active[t].box, detBoxes[d]), t, d]);
                    }
                }
                pairs.sort((p1, p2) => p2[0] - p1[0]);

                const matchedT = new Set<number>();
                const matchedD = new Set<number>();
                for (const [score, t, d] of pairs) {
                    if (score < MATCH_IOU) break;
                    if (matchedT.has(t) || matchedD.has(d)) continue;
                    matchedT.add(t);
                    matchedD.add(d);

                    const trk = active[t];
                    const box = detBoxes[d];
                    const dt = frame - trk.lastObsFrame;
                    if (dt > 0) {
                        for (let k = 0; k < 4; k++) {
                            trk.velocity[k] = 0.5 * trk.velocity[k] + (0.5 * (box[k] - trk.lastObsBox[k])) / dt;
                        }
                    }
                    trk.box = [...box];
                    trk.lastObsBox = [...box];
                    trk.lastObsFrame = frame;
                    trk.timeSinceUpdate = 0;
                    trk.observations.set(frame, dets[d]);
                }

                // unmatched detections spawn new tracks
                for (let d = 0; d < dets.length; d++) {
                    if (matchedD.has(d)) continue;
                    const box = detBoxes[d];
                    active.push({
                        box: [...box],
                        velocity: [0, 0, 0, 0],
                        lastObsBox: [...box],
                        lastObsFrame: frame,
                        timeSinceUpdate: 0,
                        labelID,
                        observations: new Map([[frame, dets[d]]]),
                    });
                }

                // retire tracks that coasted too long
                const stillActive: InternalTrack[] = [];
                for (const trk of active) {
                    (trk.timeSinceUpdate <= MAX_AGE ? stillActive : finishedTracks).push(trk);
                }
                active = stillActive;
            }

            finishedTracks.push(...active);
            processedLabels += 1;
            onProgress('Associating detections into tracks', Math.round((processedLabels / byLabel.size) * 100));
        }

        // build serialized tracks from the associations
        const tracks: SerializedTrack[] = [];
        for (const trk of finishedTracks) {
            const obsFrames = Array.from(trk.observations.keys()).sort((a, b) => a - b);
            if (obsFrames.length < MIN_HITS) {
                continue;
            }

            const firstShape = trk.observations.get(obsFrames[0]) as SerializedShape;
            const { immutable } = this.splitAttributes(firstShape.attributes);

            const trackShapes: SerializedTrack['shapes'] = obsFrames.map((frame) => {
                const shape = trk.observations.get(frame) as SerializedShape;
                const { mutable } = this.splitAttributes(shape.attributes);
                return {
                    attributes: mutable,
                    points: shape.points,
                    frame,
                    occluded: shape.occluded,
                    outside: false,
                    rotation: shape.rotation ?? 0,
                    type: ShapeType.RECTANGLE,
                    z_order: shape.z_order ?? 0,
                };
            });

            // terminate the track one frame after its last observation
            const lastFrame = obsFrames[obsFrames.length - 1];
            if (lastFrame + 1 <= maxFrame) {
                const lastShape = trk.observations.get(lastFrame) as SerializedShape;
                trackShapes.push({
                    attributes: [],
                    points: lastShape.points,
                    frame: lastFrame + 1,
                    occluded: lastShape.occluded,
                    outside: true,
                    rotation: lastShape.rotation ?? 0,
                    type: ShapeType.RECTANGLE,
                    z_order: lastShape.z_order ?? 0,
                });
            }

            tracks.push({
                label_id: trk.labelID,
                group: 0,
                frame: obsFrames[0],
                source: Source.AUTO,
                attributes: immutable,
                shapes: trackShapes,
                elements: [],
            });
        }

        return {
            created: { shapes: [], tags: [], tracks },
            deleted: { shapes: rectangles, tags: [], tracks: [] },
        };
    }

    public applyFilter(input: CollectionActionInput): CollectionActionInput['collection'] {
        return {
            shapes: input.collection.shapes.filter((shape) => shape.type === ShapeType.RECTANGLE),
            tags: [],
            tracks: [],
        };
    }

    public isApplicableForObject(objectState: ObjectState): boolean {
        return objectState.objectType === ObjectType.SHAPE && objectState.shapeType === ShapeType.RECTANGLE;
    }

    public get name(): string {
        return 'Detections to tracks';
    }

    public get parameters(): ActionParameters | null {
        return null;
    }
}
