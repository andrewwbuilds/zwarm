export function VideoFeed({ src }: { src: string }) {
  return (
    <img
      src={src}
      alt="Drone video feed"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }}
    />
  );
}
