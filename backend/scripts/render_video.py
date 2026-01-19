from moviepy import ImageClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip
import moviepy.video.fx as vfx
import os
import glob

# Configuration
SLIDES_DIR = "presentation_slides"
OUTPUT_VIDEO = "Sidecut_Doctoral_Defense_Dynamic.mp4"
SLIDE_DURATION = 10 # Increased duration for reading dense text
TRANSITION_DURATION = 1.5

def ken_burns_effect(clip, zoom_ratio=1.04):
    """
    Simulates a Ken Burns effect by slowly cropping and resizing the clip over time.
    MoviePy v2 doesn't have a built-in simple ken burns, so we simulate it with Resize.
    This is computationally expensive, so we'll do a simple constant zoom which is easier.
    Actually, correct way is: clip.with_effects([vfx.Resize(lambda t: 1 + 0.04*t/duration)])
    """
    duration = clip.duration
    # Resize function: starts at 1.0, ends at zoom_ratio
    return clip.with_effects([
        vfx.Resize(lambda t: 1 + (zoom_ratio - 1) * t / duration)
    ])

def render_video():
    print("Gathering dense slides...")
    slide_files = sorted(glob.glob(os.path.join(SLIDES_DIR, "*.png")))
    
    if not slide_files:
        print("No slides found!")
        return

    clips = []
    print(f"Found {len(slide_files)} slides. Applying cinematic efffects...")

    for i, slide_path in enumerate(slide_files):
        # Create Image Clip
        img_clip = ImageClip(slide_path).with_duration(SLIDE_DURATION)
        
        # Apply Ken Burns (Zoom In)
        # We start at 100% scale and zoom to 105% over the duration
        anim_clip = ken_burns_effect(img_clip, zoom_ratio=1.05)
        
        # Add Crossfade transition
        if i > 0:
            anim_clip = anim_clip.with_effects([vfx.CrossFadeIn(TRANSITION_DURATION)])
        
        clips.append(anim_clip)

    print("Concatenating and Rendering (This will take longer due to resizing)...")
    # Method="compose" is required for effects to blend properly
    final_video = concatenate_videoclips(clips, method="compose", padding=-TRANSITION_DURATION)

    final_video.write_videofile(OUTPUT_VIDEO, fps=24, codec='libx264')
    print(f"Done! Dynamic video saved to: {os.path.abspath(OUTPUT_VIDEO)}")

if __name__ == "__main__":
    render_video()
