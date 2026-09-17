// Writes one grayscale mask per person instance (full image size): <outDir>/<stem>-m<N>.png
// Uses Apple Vision VNGeneratePersonInstanceMaskRequest (macOS 14+).
// Note: ctx.createCGImage(format: .L8, colorSpace: nil) crashes (exit 133); write PNGs via the context instead.
import Foundation
import Vision
import CoreImage

let args = CommandLine.arguments
guard args.count == 3 else {
  FileHandle.standardError.write("usage: inst <image> <outDir>\n".data(using: .utf8)!)
  exit(2)
}
let input = URL(fileURLWithPath: args[1]), outDir = URL(fileURLWithPath: args[2])
guard let src = CIImage(contentsOf: input, options: [.applyOrientationProperty: true]) else { exit(1) }
let handler = VNImageRequestHandler(ciImage: src)
let req = VNGeneratePersonInstanceMaskRequest()
try handler.perform([req])
guard let obs = req.results?.first else { exit(0) }  // no people: write nothing
let ctx = CIContext()
let stem = input.deletingPathExtension().lastPathComponent
for i in obs.allInstances {
  let buf = try obs.generateScaledMaskForImage(forInstances: IndexSet(integer: i), from: handler)
  let m = CIImage(cvPixelBuffer: buf)
  try ctx.writePNGRepresentation(of: m, to: outDir.appendingPathComponent("\(stem)-m\(i).png"),
                                 format: .L8, colorSpace: CGColorSpace(name: CGColorSpace.linearGray)!)
}
