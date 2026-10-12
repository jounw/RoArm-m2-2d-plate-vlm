"""RoArm-M2 pick-and-place: VLM + eye-in-hand camera ONLY (no LiDAR at all).

The idea: call the VLM exactly once per cycle to find the target and hand its
box to the local tracker (ViT-Track). From then on, NOTHING re-detects or
re-draws a box during the approach -- the tracker follows the object through
every frame of the descent purely from the eye-in-hand video, all the way to
grab height. No LiDAR, no periodic VLM re-anchoring, no manual re-drawing
when getting close to the object.

This is a thin wrapper around pick_place.py's existing track_and_descend loop
(look-then-move: project the tracked box's pixel offset onto the floor plane,
correct x/y, step z down), just pinned to the settings that make "one VLM
call, then pure tracking" true:
  - --detector vlm (or --detector manual to hand-draw the one initial box
    yourself, still with zero API calls)
  - --vlm-reanchor-every 0 -- never re-call the VLM mid-approach; if the
    tracker loses the object it retries locally first and only re-calls the
    VLM as a last resort (track-lost-retries), it does not re-trigger on
    every step and never asks the user to redraw anything

Run:
  python3 pick_place_eih.py --target "the red block" --serial-port /dev/cu.usbserial-120
  python3 pick_place_eih.py --detector manual --serial-port /dev/cu.usbserial-120   # no API calls at all
"""

from __future__ import annotations

from pick_place import build_parser, main as pick_place_main


def build_eih_parser():
    parser = build_parser()
    parser.description = "RoArm-M2 pick-and-place: VLM + eye-in-hand only (no LiDAR)"
    # Pin the defaults that make this "grab the box once, then pure tracking":
    # never periodically re-call the VLM to re-anchor mid-approach.
    for action in parser._actions:
        if action.dest == "vlm_reanchor_every":
            action.default = 0
        if action.dest == "detector":
            action.default = "vlm"
    return parser


def main() -> int:
    import pick_place

    pick_place.build_parser = build_eih_parser
    return pick_place_main()


if __name__ == "__main__":
    raise SystemExit(main())
