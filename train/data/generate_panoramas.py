import os
import random
import numpy as np
from PIL import Image
import prior
import py360convert
from ai2thor.platform import CloudRendering
from ai2thor.controller import Controller

# --- THE MONKEY PATCH ---
np.bool = bool
np.float = float 
# ------------------------

os.makedirs("train/dataset/rgb", exist_ok=True)
os.makedirs("train/dataset/depth", exist_ok=True)

def generate_procthor_panoramas(num_images=5):
    print("1. Loading ProcTHOR Dataset...")
    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][0] 

    print("2. Initializing AI2-THOR Controller...")
    controller = Controller(
        agentMode="default",
        visibilityDistance=100.0,
        scene=house,
        renderDepthImage=True,
        width=512,  
        height=512,
        fieldOfView=90,
        platform=CloudRendering 
    )

    # 3. Explicitly map faces to Unity rotations
    rotations = {
        'F': dict(x=0, y=0, z=0),     # Front
        'R': dict(x=0, y=90, z=0),    # Right
        'B': dict(x=0, y=180, z=0),   # Back
        'L': dict(x=0, y=270, z=0),   # Left
        'U': dict(x=-90, y=0, z=0),   # Up
        'D': dict(x=90, y=0, z=0)     # Down
    }

    # Spawn just ONE camera to avoid AI2-THOR multi-camera sync offsets
    print("3. Spawning 360 Camera...")
    controller.step(
        action="AddThirdPartyCamera", 
        position=dict(x=0, y=0, z=0), 
        rotation=dict(x=0, y=0, z=0), 
        fieldOfView=90
    )

    positions = controller.step("GetReachablePositions").metadata["actionReturn"]

    print("4. Capturing Panoramas...")
    for i in range(num_images):
        pos = random.choice(positions)
        event = controller.step("Teleport", position=pos, rotation=dict(x=0, y=0, z=0))
        head_pos = event.metadata["cameraPosition"]

        # Teleport the agent away so it doesn't photobomb
        away_pos = random.choice(positions)
        controller.step("Teleport", position=away_pos, rotation=dict(x=0, y=0, z=0))

        cubemap_rgb = {}
        cubemap_depth = {}

        # Rotate the single camera 6 times to capture a perfect sphere
        for face, rot in rotations.items():
            event = controller.step(
                action="UpdateThirdPartyCamera", 
                thirdPartyCameraId=0, # Only use camera ID 0
                position=head_pos, 
                rotation=rot,
                fieldOfView=90 
            )
            
            # Store straight into the dictionary
            # cubemap_rgb[face] = event.third_party_camera_frames[0]
            # cubemap_depth[face] = event.third_party_depth_frames[0]

            # The [..., :3] slices off the 4th Alpha channel, converting RGBA to RGB
            cubemap_rgb[face] = event.third_party_camera_frames[0][..., :3]
            cubemap_depth[face] = event.third_party_depth_frames[0]


        # 5. Build the "Dice" (Cross) layout manually to bypass library bugs
        # Layout looks like this:
        #     [U]
        # [L] [F] [R] [B]
        #     [D]
        
        face_size = 512
        # Create empty canvas (3 rows, 4 columns)
        dice_rgb = np.zeros((face_size * 3, face_size * 4, 3), dtype=np.uint8)
        
        # Place faces in the exact "dice" positions expected by py360convert
        dice_rgb[0:face_size, face_size:face_size*2] = cubemap_rgb['U']            # Top row
        dice_rgb[face_size:face_size*2, 0:face_size] = cubemap_rgb['L']            # Mid row
        dice_rgb[face_size:face_size*2, face_size:face_size*2] = cubemap_rgb['F']  # Mid row
        dice_rgb[face_size:face_size*2, face_size*2:face_size*3] = cubemap_rgb['R']# Mid row
        dice_rgb[face_size:face_size*2, face_size*3:face_size*4] = cubemap_rgb['B']# Mid row
        dice_rgb[face_size*2:face_size*3, face_size:face_size*2] = cubemap_rgb['D']# Bottom row

        # Convert RGB
        pano_rgb = py360convert.c2e(dice_rgb, h=512, w=1024, cube_format='dice')

        # Repeat for Depth (using a single channel canvas)
        dice_depth = np.zeros((face_size * 3, face_size * 4, 1), dtype=np.float32)
        dice_depth[0:face_size, face_size:face_size*2, 0] = cubemap_depth['U']
        dice_depth[face_size:face_size*2, 0:face_size, 0] = cubemap_depth['L']
        dice_depth[face_size:face_size*2, face_size:face_size*2, 0] = cubemap_depth['F']
        dice_depth[face_size:face_size*2, face_size*2:face_size*3, 0] = cubemap_depth['R']
        dice_depth[face_size:face_size*2, face_size*3:face_size*4, 0] = cubemap_depth['B']
        dice_depth[face_size*2:face_size*3, face_size:face_size*2, 0] = cubemap_depth['D']

        pano_depth = py360convert.c2e(dice_depth, h=512, w=1024, cube_format='dice')
        pano_depth = np.squeeze(pano_depth, axis=-1)

        # DEBUG: Save the raw unfolded dice to ensure AI2-THOR aligned perfectly
        # Image.fromarray(dice_rgb).save(f"train/dataset/rgb/debug_dice_{i:04d}.jpg")

        # 6. Save to Disk
        rgb_filename = f"train/dataset/rgb/pano_{i:04d}.jpg"
        depth_filename = f"train/dataset/depth/pano_{i:04d}.npy"
        
        pano_rgb_clean = np.clip(pano_rgb, 0, 255).astype(np.uint8)
        Image.fromarray(pano_rgb_clean).convert('RGB').save(rgb_filename, quality=95)
        np.save(depth_filename, pano_depth) 
        
        print(f"   Saved {rgb_filename} and {depth_filename}")

    controller.stop()
    print("Done!")

if __name__ == "__main__":
    generate_procthor_panoramas(5)