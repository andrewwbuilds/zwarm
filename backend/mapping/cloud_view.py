"""A single-file web page that shows a point cloud you can orbit with the mouse."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

PAGE = """<!doctype html><meta charset=utf-8><title>Point cloud</title>
<style>html,body{margin:0;height:100%;background:#111;color:#ccc;font:13px system-ui}#h{position:fixed;left:12px;top:10px}</style>
<div id=h>__COUNT__ points &middot; drag to orbit, scroll to zoom, right-drag to pan</div>
<script type=importmap>{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const bytes = Uint8Array.from(atob('__DATA__'), c => c.charCodeAt(0));
const n = __COUNT__, pos = new Float32Array(bytes.buffer, 0, n * 3), rgb = new Uint8Array(bytes.buffer, n * 12, n * 3);
const colors = new Float32Array(n * 3); for (let i = 0; i < n * 3; i++) colors[i] = rgb[i] / 255;
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
geo.computeBoundingSphere();
const scene = new THREE.Scene();
const points = new THREE.Points(geo, new THREE.PointsMaterial({size: 0.03 * geo.boundingSphere.radius / 3, vertexColors: true, sizeAttenuation: true}));
scene.add(points);
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 1000);
const r = geo.boundingSphere.radius, c = geo.boundingSphere.center;
camera.up.set(0, -1, 0);  // COLMAP's y axis points down
camera.position.set(c.x, c.y - r * 0.3, c.z - r * 2.2);
const renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setSize(innerWidth, innerHeight); document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement); controls.target.copy(c);
addEventListener('resize', () => {camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight);});
renderer.setAnimationLoop(() => {controls.update(); renderer.render(scene, camera);});
</script>"""


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions and colours from a PLY written by mapping.pointcloud.write_ply."""
    data = path.read_bytes()
    header, body = data.split(b"end_header\n", 1)
    count = int(header.split(b"element vertex ")[1].split()[0])
    record = np.frombuffer(body, dtype=[("p", "<f4", 3), ("c", "u1", 3)], count=count)
    return record["p"].copy(), record["c"].copy()


def write_viewer(ply: Path, html: Path, z_up: bool = False) -> int:
    """z_up is for a map already levelled so +z is up, rather than in COLMAP's own axes."""
    xyz, rgb = read_ply(ply)
    blob = xyz.astype("<f4").tobytes() + rgb.astype("u1").tobytes()
    page = PAGE
    if z_up:
        page = page.replace("camera.up.set(0, -1, 0);  // COLMAP's y axis points down", "camera.up.set(0, 0, 1);")
        page = page.replace("camera.position.set(c.x, c.y - r * 0.3, c.z - r * 2.2);", "camera.position.set(c.x - r * 1.6, c.y, c.z + r * 1.4);")
    html.write_text(page.replace("__COUNT__", str(len(xyz))).replace("__DATA__", base64.b64encode(blob).decode()))
    return len(xyz)
