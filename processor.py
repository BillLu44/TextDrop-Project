import cv2
import numpy as np
# import imageio


def board_extractor(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Gaussian blur with kernel size 17x17
    blurred = cv2.GaussianBlur(image, (13, 13), 0)

    # Define the target color in RGB (OpenCV uses BGR, so reverse it)
    target_color = np.array([60, 70, 60], dtype=np.float32)

    # Calculate the RMS proximity for each pixel
    diff = blurred.astype(np.float32) - target_color
    rms_proximity = np.sqrt(np.sum(diff ** 2, axis=2))

    # Normalize RMS proximity to range 0-255
    rms_normalized = cv2.normalize(rms_proximity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Canny edge detection
    edges = cv2.Canny(rms_normalized, threshold1=50, threshold2=150)

    # Gaussian blur with kernel size 17x17 on edges
    blurred_edges = cv2.GaussianBlur(edges, (17, 17), 0)

    # Binary threshold with a low value of 4
    _, binary_original = cv2.threshold(blurred_edges, 4, 255, cv2.THRESH_BINARY)

    # Prepare flood fill mask and other variables
    h, w = binary_original.shape
    margin_y = h // 5  # 20% margin vertically
    margin_x = w // 5  # 20% margin horizontally
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask needs to be larger by 2 in each dimension
    max_contour = None
    max_area = 0
    # max_fill_position = None

    # Parameters for flood filling
    step = int((h * w) ** .5 // 20)  # Test every 200 pixels
    # save_gif = False  # Set to True to save a GIF of the flood fill progress
    # fps = 200  # Frames per second
    # frames = []  # List to store frames for GIF

    for y in range(margin_y, h - margin_y, step):  # Ignore top/bottom 20%
        for x in range(margin_x, w - margin_x, step):  # Ignore left/right 20%
            # Reset binary for every iteration to avoid modification issues
            binary = binary_original.copy()

            # Check conditions
            is_binary_nonzero = 1 if binary[y, x] != 255 else 0
            is_not_edge = 1 if edges[y, x] == 0 else 0
            is_similar_color = 1 if np.sqrt(np.sum((blurred[y, x].astype(np.float32) - target_color) ** 2)) < 25 else 0

            # Print conditions and current iteration
            # print(
            #     f"y: {y}, x: {x}, Conditions -> Binary: {is_binary_nonzero}, Edge: {is_not_edge}, Similar: {is_similar_color}")

            # Perform flood fill and measure area
            flood_fill_mask = mask.copy()
            cv2.floodFill(binary, flood_fill_mask, (x, y), 128)  # Use 128 as a marker
            filled_region = (flood_fill_mask[1:-1, 1:-1] == 1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(filled_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.contourArea(contours[0]) if contours else 0

            # Ensure average pixel in area is still similar to target color
            # Calculate average color in the masked region
            avg_color = cv2.mean(image, mask=filled_region)
            is_similar_color = 1 if np.sqrt(np.sum((avg_color[:3] - target_color) ** 2)) < 85 else 0

            # Create debug frame from binary (after flood fill)
            debug_frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            # Draw a circle at the current flood fill starting point
            cv2.circle(debug_frame, (x, y), 5, (0, 0, 255), -1)

            # Evaluate if all checks passed
            all_checks_passed = is_binary_nonzero and is_not_edge and is_similar_color

            # Draw current contour in blue if checks passed, red otherwise
            if contours:
                contour = max(contours, key=cv2.contourArea)
                color = (255, 0, 0) if all_checks_passed else (0, 0, 255)  # Blue for passed, red for failed
                cv2.drawContours(debug_frame, [contour], -1, color, 2)

                # Update max contour if all checks passed and the area is larger
                if all_checks_passed and area > max_area:
                    max_area = area
                    max_contour = contour
                    max_fill_position = (x, y)

            # Draw current largest contour in green
            if max_contour is not None:
                cv2.drawContours(debug_frame, [max_contour], -1, (0, 255, 0), 2)  # Green for largest contour

            # # Append the current frame to frames list for GIF export
            # save_gif and frames.append(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB))
            #
            # # Show the debug frame with updated binary and contours
            # cv2.imshow("Flood Filling Debug with Progress", debug_frame)
            #
            # # Wait for user to press a key to move to the next frame
            # key = cv2.waitKey(1000 // fps)
            # if key == ord('q'):  # Press 'q' to quit early
            #     break

    # Export frames as GIF
    # gif_path = "flood_fill_debug.gif"
    # save_gif and imageio.mimsave(gif_path, frames, fps=fps)
    # save_gif and print(f"Debug GIF saved as {gif_path}")

    # Final display
    output = image.copy()  # Copy original image for visualization

    # Draw the largest contour if found
    if max_contour is not None:
        # Draw the largest contour
        cv2.drawContours(output, [max_contour], -1, (0, 255, 0), 2)  # Green contour

        # Find extreme points on the largest contour
        rect = cv2.boundingRect(max_contour)
        x, y, w, h = rect
        box_points = [
            [x, y],  # Top-left
            [x + w, y],  # Top-right
            [x, y + h],  # Bottom-left
            [x + w, y + h],  # Bottom-right
        ]
        contour_points = np.array(max_contour).reshape(-1, 2)

        # Find the closest points on contour to the bounding box points
        src_pts = []
        for bx, by in box_points:
            distances = np.sqrt(np.sum((contour_points - np.array([bx, by])) ** 2, axis=1))
            closest_index = np.argmin(distances)
            closest_point = contour_points[closest_index]
            src_pts.append(closest_point)
            # Draw these points on the original image
            cv2.circle(output, tuple(closest_point), 5, (0, 0, 255), -1)

        # Convert source points to float32 for perspective warp
        src_pts = np.float32(src_pts)
        dst_pts = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h],
        ])

        # Compute perspective transform and warp
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (w, h))

        # Display warped image
        # cv2.imshow("Warped Perspective", warped)

        # print(f"Largest Contour Position: {max_fill_position}, Area: {max_area}")
    return warped


# Show final results
# cv2.imshow("Final Image", board_extractor("test_image/from_pi (6).jpg"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
