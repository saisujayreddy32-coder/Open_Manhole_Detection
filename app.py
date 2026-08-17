import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Open Manhole Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Open Manhole Detection")
st.write("Upload an image or video to detect manholes using YOLO.")

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --------------------------------------------------
# Select input type
# --------------------------------------------------
input_type = st.radio(
    "Select input type:",
    ["Image", "Video"],
    horizontal=True
)

# ==================================================
# IMAGE DETECTION
# ==================================================
if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Display original image
        image = Image.open(uploaded_file)

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        # Save temporary image
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp:
            temp.write(uploaded_file.getvalue())
            image_path = temp.name

        # Detect button
        if st.button("🔍 Detect Manhole"):

            with st.spinner("Detecting manhole..."):

                results = model.predict(
                    source=image_path,
                    conf=0.7
                )

                result_image = results[0].plot()

                # Convert BGR to RGB
                result_image = cv2.cvtColor(
                    result_image,
                    cv2.COLOR_BGR2RGB
                )

            st.subheader("Detection Result")
            st.image(
                result_image,
                use_container_width=True
            )

            # Detection information
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                st.success(
                    f"✅ {len(boxes)} manhole(s) detected!"
                )

                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    st.write(
                        f"**{class_name}** — "
                        f"Confidence: {confidence:.2%}"
                    )
            else:
                st.warning("⚠️ No manhole detected.")

        # Remove temporary file
        if os.path.exists(image_path):
            os.remove(image_path)


# ==================================================
# VIDEO DETECTION
# ==================================================
else:

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("🔍 Detect Manholes in Video"):

            # ------------------------------------------
            # Save uploaded video
            # ------------------------------------------
            input_path = os.path.join(
                tempfile.gettempdir(),
                uploaded_video.name
            )

            with open(input_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            # ------------------------------------------
            # Open video
            # ------------------------------------------
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("❌ Could not open the uploaded video.")
                st.stop()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            # ------------------------------------------
            # Temporary AVI output
            # ------------------------------------------
            temp_avi = os.path.join(
                tempfile.gettempdir(),
                "manhole_detected.avi"
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"XVID"
            )

            writer = cv2.VideoWriter(
                temp_avi,
                fourcc,
                fps,
                (width, height)
            )

            progress = st.progress(0)

            frame_count = 0
            detected_frames = 0

            # ------------------------------------------
            # YOLO detection
            # ------------------------------------------
            with st.spinner(
                "Detecting manholes in video..."
            ):

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.predict(
                        source=frame,
                        conf=0.7,
                        verbose=False
                    )

                    # Draw bounding boxes
                    annotated_frame = results[0].plot()

                    writer.write(
                        annotated_frame
                    )

                    # Count detections
                    if (
                        results[0].boxes is not None
                        and len(results[0].boxes) > 0
                    ):
                        detected_frames += 1

                    frame_count += 1

                    if total_frames > 0:
                        progress.progress(
                            min(
                                frame_count / total_frames,
                                1.0
                            )
                        )

            cap.release()
            writer.release()

            # ------------------------------------------
            # Convert AVI -> H.264 MP4
            # ------------------------------------------
            output_mp4 = os.path.join(
                tempfile.gettempdir(),
                "manhole_detection_result.mp4"
            )

            import subprocess

            command = [
                "ffmpeg",
                "-y",
                "-i",
                temp_avi,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_mp4
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # ------------------------------------------
            # Check conversion
            # ------------------------------------------
            if not os.path.exists(output_mp4):
                st.error("❌ Video conversion failed.")
                st.code(
                    result.stderr.decode(
                        errors="ignore"
                    )
                )
                st.stop()

            # ------------------------------------------
            # Read MP4
            # ------------------------------------------
            with open(output_mp4, "rb") as f:
                video_bytes = f.read()

            st.success(
                "✅ Manhole detection completed!"
            )

            st.subheader(
                "🎥 YOLO Detection Result"
            )

            # ------------------------------------------
            # PLAY VIDEO
            # ------------------------------------------
            st.video(
                video_bytes,
                format="video/mp4"
            )

            # ------------------------------------------
            # Download button
            # ------------------------------------------
            st.download_button(
                label="⬇️ Download Detected Video",
                data=video_bytes,
                file_name="manhole_detection_result.mp4",
                mime="video/mp4"
            )

            st.info(
                f"Frames with manhole detections: "
                f"{detected_frames}"
            )