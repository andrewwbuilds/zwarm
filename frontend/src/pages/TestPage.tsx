import StaticDrawingOverlay from "../components/StaticDrawingOverlay";
import { VideoFeed } from "../components/VideoFeed";
import { API_BASE_URL } from "../backend";

export default function TestPage() {
  return (
    <div>
      <p>static video feed, 640x480</p>
      <div style={{ position: "relative", width: 640, height: 480 }}>
        <VideoFeed src="http://localhost:8000/mjpeg" />
        <StaticDrawingOverlay />
      </div>
    </div>
  );
} 