# Dead ends

These were tried on real event photos and dropped. Don't reintroduce them without a new reason.

- **`VNGeneratePersonSegmentationRequest` alone.** Soft, mushy edges, and held cans turn
  semi-transparent. It was the whole problem of the first version.
- **`VNGenerateForegroundInstanceMaskRequest` (Subject Lifting) alone.** A seated group plus the table
  becomes one single object, so there's no way to separate people.
- **Subject Lifting multiplied by a dilated person instance mask.** Neighbours and the table still
  remain inside the mask.
- **PIL `MaxFilter` with a large radius at full resolution.** Runtime grows with the square of the radius:
  several minutes per image. Dilate at 1/8 scale instead.
- **Swift `ctx.createCGImage(..., format: .L8, colorSpace: nil)`.** Crashes with exit code 133.
  Write masks with `writePNGRepresentation` (or `render(toBitmap:)`) and an explicit grey colour space.
- **ffmpeg `drawtext` for contact sheets.** Homebrew's ffmpeg is built without it. Pillow does the job.
- **Cloud background removers** (Adobe Remove Background and similar). Ruled out for privacy.
- **An LLM looking at contact sheets to assign names.** It works, but it sends photos of people to an
  AI service. Replaced by local face clustering plus naming by the user.
- **Ultralytics YOLO segmentation / insightface.** Not used because of licensing: Ultralytics is
  AGPL-3.0, insightface's pretrained models are non-commercial only.
- **CoreML execution provider for BiRefNet.** Tested on an M2 Max; see troubleshooting for numbers.
  It failed after a 14-minute compile, so `auto` means CPU.
