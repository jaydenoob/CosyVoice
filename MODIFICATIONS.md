# Modifications

This repository is a modified version of CosyVoice. The original project is
licensed under the Apache License 2.0, and the original copyright and license
notices should be retained.

This fork focuses on local WebUI usability improvements:

- Added WebUI compatibility adjustments for CosyVoice3 0.5B local usage.
- Added segmented text generation with cumulative audio preview.
- Added progress display with completed segment count, elapsed time, remaining
  time estimate, and estimated completion time.
- Added a generation cancellation request button for stopping after the current
  segment.
- Prevented concurrent generations from starting in the same WebUI session.
- Saved generated audio to local WAV files so browser playback and download work
  reliably.
- Disabled automatic playback after completion.
- Added completion notification and a short prompt sound.
- Improved prompt-audio preparation and error handling for browser-uploaded
  files.

- Added Apple Silicon MPS inference support with CPU fallbacks for unsupported operations.
- Corrected CosyVoice 3 prompt and instruction formatting.
- Preserved accumulated streaming audio and split long unpunctuated Chinese text.

This fork is not an official CosyVoice distribution.
