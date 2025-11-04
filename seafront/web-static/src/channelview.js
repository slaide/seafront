"use strict";

import * as THREE from 'three';

/**
 * @typedef {Object} ChannelImageData
 * @property {number} width
 * @property {number} height
 * @property {Uint8Array|Uint16Array} data
 * @property {THREE.DataTexture} texture
 * @property {THREE.Mesh} mesh
 */

/**
 * @typedef {Object} SceneInfoListeners
 * @property {HTMLElement} elem
 * @property {(event: WheelEvent) => void} wheel
 * @property {(event: MouseEvent) => void} mousedown
 * @property {(event: MouseEvent) => void} mouseup
 * @property {(event: MouseEvent) => void} mousemove
 */

/**
 * @typedef {Object} SceneInfo
 * @property {string} channelhandle
 * @property {THREE.Scene} scene
 * @property {THREE.OrthographicCamera} camera
 * @property {HTMLElement|null} elem
 * @property {THREE.Mesh|undefined} mesh
 * @property {ChannelImageData|undefined} img
 * @property {{zoom:number, offsetx:number, offsety:number}} range
 * @property {SceneInfoListeners|null} listeners
 */

const delta_time = 1. / 30.
export class ChannelImageView {
    /**
     * 
     * @param {HTMLCanvasElement} canvas 
     * @param {Map<string,CachedChannelImage>} cached_channel_image
     */
    constructor(canvas, cached_channel_image) {
        this.canvas = canvas
        this.cached_channel_image = cached_channel_image

        const renderer = new THREE.WebGLRenderer({ antialias: false, canvas, alpha: true, powerPreference: "high-performance" })
        renderer.setPixelRatio(window.devicePixelRatio) // enable ssaa by *1.5 (bad for performance)
        this.renderer = renderer

        // set clear color based on theme color
        this.setClearColorFromBody();

        /** @type {SceneInfo[]} */
        this.sceneInfos = []

        this.scrollSpeed = 2e-3;
        // allow zooming in quite far, but not zoom out much (there is nothing to see outside the image)
        this.zoomLimit = { min: 0.05, max: 1.2 };

        this.drag = { active: false, x: 0, y: 0 };

        // toggle drawing loop based on visibility of the plot container
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                // entry.isIntersecting will be true if the element is visible in the viewport
                if (entry.isIntersecting) {
                    observer.unobserve(renderer.domElement)

                    this._initializeExistingChannelElements();
                    this.draw()
                }
            });
        }, {
            // Optional: adjust threshold if needed
            threshold: 0.1  // 10% visibility is enough to start the animation loop
        });

        // Begin observing the canvas element
        observer.observe(renderer.domElement);

        this.draw()
    }

    _initializeExistingChannelElements() {
        const elements = document.getElementsByClassName("channel-box-image");
        for (let el of elements) {
            if (!(el instanceof HTMLElement)) continue;
            this.ensureSceneForElement(el);
        }
    }

    setClearColorFromBody() {
        const computedStyle = getComputedStyle(document.body);
        const themeBgColor = computedStyle.getPropertyValue("--strong-bg-color").trim();
        this.renderer.setClearColor(new THREE.Color(themeBgColor || 'rgb(255,255,255)'), 1);
    }

    /** Update theme colors - call this when theme changes */
    updateTheme() {
        this.setClearColorFromBody();
    }

    /**
     * get bounding box of element to draw canvas in
     * @returns {DOMRect}
     */
    getRect() {
        const parent = this.renderer.domElement.parentElement
        if (!parent) throw new Error(`parent is undefined`);
        let parentRect = parent.getBoundingClientRect()
        return parentRect
    }

    /**
     * 
     * @param {THREE.WebGLRenderer} renderer 
     * @returns 
     */
    resizeRendererToDisplaySize(renderer) {
        const rect = this.getRect();
        const width = rect.width;
        const height = rect.height;

        const canvas = renderer.domElement.getBoundingClientRect()

        const needResize = canvas.width !== width || canvas.height !== height;
        if (needResize) {
            // console.log("set canvas size to",rect)
            renderer.setSize(width, height, false);
            renderer.domElement.style["width"] = `${width}px`
            renderer.domElement.style["height"] = `${height}px`
            renderer.domElement.style["top"] = `${rect.y}px`
            renderer.domElement.style["left"] = `${rect.x}px`
        }

        return needResize;
    }

    /**
     * 
     * @param {HTMLElement} elem 
     * @returns {SceneInfo}
     */
    _makeImageScene(elem) {
        const channelhandle = elem.parentElement?.getAttribute(`channelhandle`)
        if (!channelhandle) { const error = `${elem} has no attribute "channelhandle"`; console.error(error); throw error }

        const scene = new THREE.Scene();

        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1, 1);
        camera.position.z = 1;

        /** @type {SceneInfo} */
        const sceneInfo = {
            channelhandle, scene, camera, elem,
            mesh: undefined, img: undefined,
            range: { zoom: 1, offsetx: 0, offsety: 0 },
            listeners: null
        };

        let imageinfo = this.cached_channel_image.get(channelhandle)
        if (imageinfo) {
            this.updateTextureData(sceneInfo, imageinfo);
        }

        return sceneInfo;
    }

    /**
     * Ensure we have a scene for the given element and attach interactions.
     * @param {HTMLElement} elem
     * @returns {SceneInfo}
     */
    ensureSceneForElement(elem) {
        const channelhandle = elem.parentElement?.getAttribute(`channelhandle`);
        if (!channelhandle) {
            const error = `${elem} has no attribute "channelhandle"`;
            console.error(error);
            throw error;
        }

        let sceneInfo = this.sceneInfos.find(info => info.channelhandle === channelhandle);

        if (!sceneInfo) {
            sceneInfo = this._makeImageScene(elem);
            this.sceneInfos.push(sceneInfo);
        } else {
            sceneInfo.elem = elem;
        }

        this._attachInteractionHandlers(sceneInfo);

        const cachedImage = this.cached_channel_image.get(channelhandle);
        if (cachedImage) {
            this.updateTextureData(sceneInfo, cachedImage);
        }

        return sceneInfo;
    }

    /**
     * Attach zoom/drag interactions to the scene element, replacing previous bindings if needed.
     * @param {SceneInfo} sceneInfo
     */
    _attachInteractionHandlers(sceneInfo) {
        const elem = sceneInfo.elem;
        if (!elem) return;

        const existing = sceneInfo.listeners;
        if (existing && existing.elem !== elem && existing.elem instanceof HTMLElement) {
            existing.elem.removeEventListener("wheel", existing.wheel, true);
            existing.elem.removeEventListener("mousedown", existing.mousedown);
            existing.elem.removeEventListener("mouseup", existing.mouseup);
            existing.elem.removeEventListener("mousemove", existing.mousemove);
            sceneInfo.listeners = null;
        } else if (existing && existing.elem === elem) {
            return;
        }

        const wheelHandler = /** @type {(event: WheelEvent) => void} */((event) => {
            event.preventDefault();

            // calculate zoom factor
            let delta_zoom = event.deltaY * this.scrollSpeed;
            // apply
            if (delta_zoom > 0) {
                sceneInfo.range.zoom *= 1 + delta_zoom;
            } else {
                sceneInfo.range.zoom /= 1 - delta_zoom;
            }

            sceneInfo.range.zoom = Math.min(Math.max(sceneInfo.range.zoom, this.zoomLimit.min), this.zoomLimit.max);

            // link zooms across channel views
            for (const otherscene of this.sceneInfos) {
                otherscene.range.zoom = sceneInfo.range.zoom;
            }

        });

        const mouseDownHandler = /** @type {(event: MouseEvent) => void} */((event) => {
            event.preventDefault();
            this.drag.active = true;
            this.drag.x = event.clientX;
            this.drag.y = event.clientY;
        });

        const mouseUpHandler = /** @type {(event: MouseEvent) => void} */((event) => {
            event.preventDefault();
            this.drag.active = false;
        });

        const mouseMoveHandler = /** @type {(event: MouseEvent) => void} */((event) => {
            if (!this.drag.active) return;

            event.preventDefault();

            // calculate offset in screen pixels
            const deltax = event.clientX - this.drag.x;
            const deltay = event.clientY - this.drag.y;

            // Get element dimensions to calculate pixel-to-image coordinate conversion
            const elemRect = elem.getBoundingClientRect();

            if (sceneInfo.img && elemRect.width > 0 && elemRect.height > 0) {
                // Calculate the actual displayed image size accounting for zoom and aspect ratio
                const displayedImageWidth = sceneInfo.img.width * sceneInfo.range.zoom;
                const displayedImageHeight = sceneInfo.img.height * sceneInfo.range.zoom;

                // Calculate the scale factor from screen pixels to image coordinates
                // This accounts for how much of the element is filled by the image
                const scaleX = displayedImageWidth / elemRect.width;
                const scaleY = displayedImageHeight / elemRect.height;

                // Apply the drag with proper scaling so mouse stays pinned to same image pixel
                sceneInfo.range.offsetx -= deltax * scaleX;
                sceneInfo.range.offsety += deltay * scaleY;
            } else {
                // Fallback to old behavior if image not loaded yet
                sceneInfo.range.offsetx -= deltax * sceneInfo.range.zoom;
                sceneInfo.range.offsety += deltay * sceneInfo.range.zoom;
            }

            // update current cursor position for later updates
            this.drag.x = event.clientX;
            this.drag.y = event.clientY;

            // link offsets across channel views
            for (const otherscene of this.sceneInfos) {
                otherscene.range.offsetx = sceneInfo.range.offsetx;
                otherscene.range.offsety = sceneInfo.range.offsety;
            }
        });

        elem.addEventListener("wheel", wheelHandler, { capture: true, passive: false });
        elem.addEventListener("mousedown", mouseDownHandler);
        elem.addEventListener("mouseup", mouseUpHandler);
        elem.addEventListener("mousemove", mouseMoveHandler);

        sceneInfo.listeners = {
            elem,
            wheel: wheelHandler,
            mousedown: mouseDownHandler,
            mouseup: mouseUpHandler,
            mousemove: mouseMoveHandler,
        };
    }

    /**
     * 
     * @param {CachedChannelImage} imageinfo 
     * @returns {{imgdata:Uint16Array|Uint8Array,datatype:THREE.UnsignedByteType|THREE.UnsignedShortType}}
     */
    _imageInfoToImage(imageinfo) {

        // 2. Create a DataTexture using the LuminanceFormat to preserve the single-channel data.
        let datatype
        let imgdata
        switch (imageinfo.bit_depth) {
            case 8: {
                datatype = THREE.UnsignedByteType;
                imgdata = new Uint8Array(imageinfo.data);
                break;
            }
            case 16: {
                datatype = THREE.UnsignedShortType;
                imgdata = new Uint16Array(imageinfo.data);
                break;
            }
            default:
                throw new Error(`unknown bitdepth ${imageinfo.bit_depth} ${Array.from(Object.keys(imageinfo))}`);
        }

        return { imgdata, datatype }
    }

    /**
     * @param {CachedChannelImage} imageinfo
     * @returns {ChannelImageData}
     */
    _makeImage(imageinfo) {
        const WIDTH_DOWNSAMPLE_FACTOR = 2
        const HEIGHT_DOWNSAMPLE_FACTOR = 2

        let { imgdata, datatype } = this._imageInfoToImage(imageinfo)

        const width = imageinfo.width / WIDTH_DOWNSAMPLE_FACTOR
        const height = imageinfo.height / HEIGHT_DOWNSAMPLE_FACTOR
        const data = imgdata

        const texture = new THREE.DataTexture(
            imgdata,
            width,
            height,
            // single color channel (just defaults to red)
            // also, (unsigned) integer values
            THREE.RedIntegerFormat,
            datatype
        );
        texture.minFilter = THREE.NearestFilter;
        texture.magFilter = THREE.NearestFilter;
        texture.needsUpdate = true;

        // 3. Write custom shaders. The vertex shader passes the UV coordinates,
        // and the fragment shader reads the single-channel texture and outputs a grayscale color.
        const vertexShader = `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            // sample integer image
            uniform usampler2D uTexture;

            varying vec2 vUv;
            void main() {
                // sample from texture
                // red channel (based on format)
                // convert int value to float
                // Invert Y-coordinate to compensate for GPU coordinate system (Y-up)
                // vs numpy array coordinate system (Y-down, origin top-left)
                float lum = float(texture2D(uTexture, vec2(vUv.x, 1.0 - vUv.y)).r);

                // adjust from [0;formatMax] to [0;1] space
                lum/=float(1<<(${imageinfo.bit_depth}));

                gl_FragColor = vec4(vec3(lum), 1.0);
            }
        `;

        // 4. Create a ShaderMaterial using the custom shaders and pass the texture via a uniform.
        const material = new THREE.ShaderMaterial({
            uniforms: {
                uTexture: { value: texture }
            },
            vertexShader,
            fragmentShader
        });

        // 5. Create a simple quad (a PlaneGeometry) on which the texture is drawn.
        const geometry = new THREE.PlaneGeometry(width, height);
        const mesh = new THREE.Mesh(geometry, material);

        /** @type {ChannelImageData} */
        const img = {
            width: imageinfo.width / WIDTH_DOWNSAMPLE_FACTOR,
            height: imageinfo.height / HEIGHT_DOWNSAMPLE_FACTOR,
            data: imgdata,
            texture,
            mesh,
        }

        return img
    }

    /**
     * 
     * @param {THREE.OrthographicCamera} camera 
     * @param {{width:number,height:number,center:{x:number,y:number},aspect_ratio?:number}} opt 
     */
    cameraFit(camera, opt) {
        let { width, height, center } = opt
        let target_aspect_ratio = opt.aspect_ratio ?? 1

        let current_aspect_ratio = width / height
        if (current_aspect_ratio > target_aspect_ratio) {
            height = height * current_aspect_ratio / target_aspect_ratio
        } else {
            width = width / current_aspect_ratio * target_aspect_ratio
        }

        camera.left = center.x - width / 2
        camera.right = center.x + width / 2
        camera.top = center.y + height / 2
        camera.bottom = center.y - height / 2
        //console.log("set camera bounding box to", center.x - width / 2, center.x + width / 2, center.y + height / 2, center.y - height / 2)
    }

    /**
     * Function to update the texture with 8 random byte values
     * 
     * @param {SceneInfo} sceneInfo
     * @param {CachedChannelImage} newimageinfo
     */
    updateTextureData(sceneInfo, newimageinfo) {
        let { imgdata, datatype } = this._imageInfoToImage(newimageinfo)

        const channelhandle = newimageinfo.info.channel.handle
        if ((sceneInfo.elem?.getBoundingClientRect().width ?? 0) == 0) {
            const new_element = Array.from(document.getElementsByClassName("channel-box-image")).find(
                e => e.parentElement?.getAttribute(`channelhandle`) == channelhandle
            )

            if (!(new_element instanceof HTMLElement)) { throw new Error(`element not found or invalid ${new_element}`); }
            sceneInfo.elem = new_element
        }

        // Check if we need to recreate the image due to bit depth change
        const needsRecreation = !sceneInfo.img || 
            (sceneInfo.img.texture.type !== datatype) ||
            (sceneInfo.img.texture.image.data.constructor !== imgdata.constructor);

        if (needsRecreation) {
            // Remove old mesh from scene if it exists
            if (sceneInfo.img && sceneInfo.mesh) {
                sceneInfo.scene.remove(sceneInfo.mesh);
                sceneInfo.img.texture.dispose();
                sceneInfo.img.mesh.geometry.dispose();
                sceneInfo.img.mesh.material.dispose();
            }

            // Create new image with correct format
            sceneInfo.img = this._makeImage(newimageinfo);

            if (sceneInfo.img) {
                sceneInfo.scene.add(sceneInfo.img.mesh);
            }

            sceneInfo.mesh = sceneInfo.img?.mesh;
        } else {
            // Safe to update existing texture since formats match
            if (!sceneInfo.img) { const error = `sceneInfo.img is null`; console.error(error); throw error }

            const texture = sceneInfo.img.texture
            if (!texture) { const error = `texture is null`; console.error(error); throw error }
            if (texture.image.data.length != imgdata.length) {
                console.error("length does not match", texture.image.data.length, imgdata.length)
                return
            }
            texture.image.data = imgdata
            texture.needsUpdate = true; // signal Three.js that the texture data has changed
        }
    }

    /**
     * render one channel
     * @param {SceneInfo} sceneInfo 
     * @returns 
     */
    renderSceneInfo(sceneInfo) {
        const { channelhandle, scene, camera } = sceneInfo;

        const elem = document.getElementById(`channelview-item-${channelhandle}`);
        if (!elem) {
            // console.error(`could not element with id 'channelview-item-${channelhandle}'`);
            return;
        }

        // get the viewport relative position of this element
        const { left, right, top, bottom, width, height } =
            elem.getBoundingClientRect();

        const isOffscreen =
            bottom < 0 ||
            top > this.getRect().height ||
            right < 0 ||
            left > this.getRect().width;

        if (isOffscreen) {
            return;
        }

        if (sceneInfo.img?.texture) {
            this.cameraFit(camera, {
                height: sceneInfo.img.height * sceneInfo.range.zoom,
                width: sceneInfo.img.width * sceneInfo.range.zoom,
                center: {
                    x: sceneInfo.range.offsetx,
                    y: sceneInfo.range.offsety,
                },
                aspect_ratio: elem.getBoundingClientRect().width / elem.getBoundingClientRect().height
            })
        } else {
            camera.aspect = width / height;
        }
        camera.updateProjectionMatrix();

        const canvasRect = this.getRect()
        const positiveYUpBottom = canvasRect.y + canvasRect.height - bottom;
        this.renderer.setScissor(left, positiveYUpBottom, width, height);
        this.renderer.setViewport(left, positiveYUpBottom, width, height);

        this.renderer.render(scene, camera);
    }

    /**
     * @param {number} time deltatime
     */
    _render(time) {
        if (this.resizeRendererToDisplaySize(this.renderer)) {
            // (resizing happens inside resizeRendererToDisplaySize, and just returns
            // true to indicate if resizing has taken place. code path here may be used
            // to check that resizing actually just happens on demand)

            // console.log("resized");
        }

        this.renderer.setScissorTest(false);
        this.renderer.clear(true, true);
        this.renderer.setScissorTest(true);

        // update texture
        // updateTextureData(sceneInfos[1].img.texture);

        for (let sceneInfo of this.sceneInfos) {
            this.renderSceneInfo(sceneInfo);
        }
    }

    /**
     * draw (will schedule itself running again later)
     */
    draw() {
        this._render(delta_time)
        requestAnimationFrame(() => this.draw())
    }
}
