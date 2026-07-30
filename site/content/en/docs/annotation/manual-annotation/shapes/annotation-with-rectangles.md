---
title: 'Annotation with rectangles'
linkTitle: 'Rectangles'
weight: 1
aliases:
- /docs/manual/advanced/annotation-with-rectangles/
- /docs/annotation/tools/annotation-with-rectangles/
---

To learn more about annotation using a rectangle, see the sections:
- {{< ilink "/docs/annotation/manual-annotation/shapes/shape-mode-basics" "Shape mode (basics)" >}}
- {{< ilink "/docs/annotation/manual-annotation/shapes/track-mode-basics" "Track mode (basics)" >}}

## Rotation rectangle

To rotate the rectangle, pull on the `rotation point`. Rotation is done around the center of the rectangle.
To rotate at a fixed angle (multiple of 15 degrees),
hold `shift`. In the process of rotation, you can see the angle of rotation.

![Annotation with rectangle shape and highlighted rotation point](/images/image230.jpg)

## Moving several rectangles at once

Hold `Ctrl` and click rectangles to add them to a selection, selected rectangles are
outlined with a dashed blue border. `Ctrl` + click a selected rectangle again to remove it
from the selection, and click anywhere aside of the objects to reset the selection completely.
To select every movable rectangle on the frame at once, press `Ctrl+Shift+A`
(the shortcut is configurable in the settings).

Once at least two rectangles are selected, dragging any of them moves the whole selection
by the same offset. The move is written to the annotations history as a single action,
so one `Ctrl+Z` returns every rectangle back.

Pressing the delete object shortcut (`Del` by default, `Shift+Del` to also remove locked
objects) while a selection exists removes every selected rectangle at once, again as a
single history action.

Locked, hidden and non-rectangular objects are not included into such a selection,
and the selection is reset when you switch to another frame.

## Annotation with rectangle by 4 points

It is an efficient method of bounding box annotation, proposed
[here](https://arxiv.org/pdf/1708.02750.pdf).
Before starting, you need to make sure that the drawing method by 4 points is selected.

![Open "Draw new rectangle" window with highlighted "By 4 points" option](/images/image134.jpg)

Press `Shape` or `Track` for entering drawing mode. Click on four extreme points:
the top, bottom, left- and right-most physical points on the object.
Drawing will be automatically completed right after clicking the fourth point.
Press `Esc` to cancel editing.

![Example of annotation process made with four point rectangle](/images/gif016_mapillary_vistas.gif)
