"use strict";

import * as THREE from 'three';
// just used for webgl availability checks
import WebGL from 'three/addons/capabilities/WebGL';
// to load a font for well texts
import { FontLoader } from 'three/addons/loaders/FontLoader';
// to merge buffer geometries to draw all well text in a single common call
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils';

/**
 * Calculate the physical position of a site within a well
 * @param {number} wellX_mm - Well X position in mm
 * @param {number} wellY_mm - Well Y position in mm  
 * @param {number} wellSizeX_mm - Well width in mm
 * @param {number} wellSizeY_mm - Well height in mm
 * @param {number} siteX - Site column index
 * @param {number} siteY - Site row index
 * @param {number} gridNumX - Total grid columns
 * @param {number} gridNumY - Total grid rows
 * @param {number} deltaX_mm - Site spacing X in mm
 * @param {number} deltaY_mm - Site spacing Y in mm
 * @param {number} fovX - Objective FOV width in mm
 * @param {number} fovY - Objective FOV height in mm
 * @returns {{x: number, y: number}} Site position in mm
 */
export function calculateSitePosition(wellX_mm, wellY_mm, wellSizeX_mm, wellSizeY_mm, 
                                     siteX, siteY, gridNumX, gridNumY, 
                                     deltaX_mm, deltaY_mm, fovX, fovY) {
    const site_plate_x_offset = wellX_mm + wellSizeX_mm / 2
        - (fovX + (gridNumX - 1) * deltaX_mm) / 2
        + siteX * deltaX_mm;
    const site_plate_y_offset = wellY_mm + wellSizeY_mm / 2
        - (fovY + (gridNumY - 1) * deltaY_mm) / 2
        + (gridNumY - 1 - siteY) * deltaY_mm;
    
    return { x: site_plate_x_offset, y: site_plate_y_offset };
}

/**
 * Calculate well position on the plate
 * @param {Wellplate} plate - Wellplate configuration
 * @param {number} x - Well column index
 * @param {number} y - Well row index
 * @returns {{x: number, y: number}} Well position in mm
 */
export function calculateWellPosition(plate, x, y) {
    const well_x = plate.Offset_A1_x_mm + x * plate.Well_distance_x_mm;
    const well_y = plate.Offset_A1_y_mm + (plate.Num_wells_y - 1 - y) * plate.Well_distance_y_mm;
    return { x: well_x, y: well_y };
}

/**
 * Calculate site position using Python protocol runner logic (for hit detection)
 * @param {number} wellX_mm - Well X position in mm
 * @param {number} wellY_mm - Well Y position in mm  
 * @param {number} wellSizeX_mm - Well width in mm
 * @param {number} wellSizeY_mm - Well height in mm
 * @param {number} siteX - Site column index
 * @param {number} siteY - Site row index
 * @param {number} gridNumX - Total grid columns
 * @param {number} gridNumY - Total grid rows
 * @param {number} deltaX_mm - Site spacing X in mm
 * @param {number} deltaY_mm - Site spacing Y in mm
 * @returns {{x: number, y: number}} Site position in mm
 */
export function calculateSitePositionPython(wellX_mm, wellY_mm, wellSizeX_mm, wellSizeY_mm, 
                                           siteX, siteY, gridNumX, gridNumY, 
                                           deltaX_mm, deltaY_mm) {
    // Python protocol runner logic: center grid within well
    const site_plate_x_offset = wellX_mm + wellSizeX_mm / 2
        - (gridNumX - 1) / 2 * deltaX_mm
        + siteX * deltaX_mm;
    const site_plate_y_offset = wellY_mm + wellSizeY_mm / 2
        - (gridNumY - 1) / 2 * deltaY_mm
        + siteY * deltaY_mm;
    
    return { x: site_plate_x_offset, y: site_plate_y_offset };
}

/**
 * Transform coordinates from backend physical system to display coordinate system
 * @param {number} x_mm - X coordinate in mm (backend coordinates)
 * @param {number} y_mm - Y coordinate in mm (backend coordinates)
 * @param {Wellplate} plate - Plate configuration for coordinate bounds
 * @returns {{x: number, y: number}} Coordinates transformed for display system
 */
export function transformBackendToDisplayCoordinates(x_mm, y_mm, plate) {
    // X coordinate stays the same
    // Y coordinate needs to be flipped: display_y = plate_height - backend_y
    return {
        x: x_mm,
        y: plate.Width_mm - y_mm
    };
}

/**
 * make well name
 *
 * Example:
 * ```js
 * makeWellName(0,0)=="A01"
 * makeWellName(1,1)=="B02"
 * makeWellName(0,10)=="A11"
 * makeWellName(26,20)=="a21"
 * ```
 * @param {number} x
 * @param {number} y
 * @returns {string}
 */
export function makeWellName(x, y) {
    let ret;

    // row index names are A...Za..z
    if (y < 26) {
        ret = String.fromCharCode("A".charCodeAt(0) + y);
    } else {
        ret = String.fromCharCode("a".charCodeAt(0) + (y - 26));
    }

    // get number as string
    const xstring = (x + 1).toString();
    const xlen = xstring.length;

    // pad with leading zero to 2 digits
    ret += "0".repeat(Math.max(0, 2 - xlen));

    ret += xstring;

    return ret;
}

/**
 * 
 * @param {AABB} aabb 
 * @param {THREE.Material} mat
 * @returns {THREE.Mesh}
 */
function makeQuad(aabb, mat) {
    let quad_shape = new THREE.Shape();
    quad_shape.moveTo(aabb.ax, aabb.ay);
    quad_shape.lineTo(aabb.ax, aabb.by);
    quad_shape.lineTo(aabb.bx, aabb.by);
    quad_shape.lineTo(aabb.bx, aabb.ay);
    let quadgeo = new THREE.ShapeGeometry(quad_shape);
    let quad = new THREE.Mesh(quadgeo, mat);
    return quad;
}

/**
 * 
 * @param {AABB} aabb 
 * @param {THREE.Material} mat
 * @param {{border_radius?:number,segments?:number}?} opts
 * @returns {THREE.Mesh}
 */
function makeRoundedQuad(aabb, mat, opts) {
    const width = aabb.bx - aabb.ax;
    const height = aabb.by - aabb.ay;

    const border_radius = opts?.border_radius ?? Math.min(width, height) / 4;
    const segments = opts?.segments ?? 4;

    if (segments < 1) throw new Error(`segments ${segments} is invalid (must be >=1)`);
    // ensure border radius makes sense
    if (border_radius < 0 || border_radius > width / 2 || border_radius > height / 2) throw new Error(`border radius ${border_radius} is out of bounds [0;min(${width / 2},${height / 2})]`);

    const geo = RoundedRectangleIndexed(width, height, border_radius, segments);

    // center of quad is at (0,0) -> adjust to aabb
    geo.translate(width / 2 + aabb.ax, height / 2 + aabb.ay, 0);

    // THREE.LineBasicMaterial is not compatible with BufferGeometry.
    // https://threejs.org/docs/#api/en/materials/MeshBasicMaterial
    let quad = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: mat.color,
        side: THREE.DoubleSide,
        transparent: true,
        wireframe: false,
        opacity: 0.5,
    }));

    return quad;

    // square with rounded edge generator implementations from https://discourse.threejs.org/t/how-to-make-rounded-edge-in-plane-geometry/64048/4

    /**
     * indexed BufferGeometry
     * @param {number} w width
     * @param {number} h height
     * @param {number} r corner radius
     * @param {number} s smoothness
     * @returns 
     */
    function RoundedRectangleIndexed(w, h, r, s) {
        const wi = w / 2 - r;   // inner width
        const hi = h / 2 - r;   // inner height
        const ul = r / w;       // u left
        const ur = (w - r) / w; // u right
        const vl = r / h;       // v low
        const vh = (h - r) / h; // v high	

        let positions = [ wi, hi, 0, -wi, hi, 0, -wi, -hi, 0, wi, -hi, 0 ];

        let uvs = [ ur, vh, ul, vh, ul, vl, ur, vl ];

        let n = [
            3 * (s + 1) + 3, 3 * (s + 1) + 4, s + 4, s + 5,
            2 * (s + 1) + 4, 2, 1, 2 * (s + 1) + 3,
            3, 4 * (s + 1) + 3, 4, 0
        ];

        let indices = [
            n[0], n[1], n[2], n[0], n[2], n[3],
            n[4], n[5], n[6], n[4], n[6], n[7],
            n[8], n[9], n[10], n[8], n[10], n[11]
        ];

        for (let i = 0; i < 4; i++) {
            const xc = i < 1 || i > 2 ? wi : -wi;
            const yc = i < 2 ? hi : -hi;

            const uc = i < 1 || i > 2 ? ur : ul;
            const vc = i < 2 ? vh : vl;

            for (let j = 0; j <= s; j++) {
                const phi = Math.PI / 2 * (i + j / s);
                const cos = Math.cos(phi);
                const sin = Math.sin(phi);

                positions.push(xc + r * cos, yc + r * sin, 0);
                uvs.push(uc + ul * cos, vc + vl * sin);

                if (j < s) {
                    const idx = (s + 1) * i + j + 4;
                    indices.push(i, idx, idx + 1);
                }
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
        geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));

        return geometry;
    }
}

let objectNamePlate = "plate";
let objectNameWells = "wells";
let objectNameWellsSelected = "wellsSelected";
let objectNameSites = "sites";
let objectNameText = "wellLabels";
let objectNameForbiddenAreas = "forbiddenAreas";
let objectNameOuterOutline = "outerOutline";
let objectNameInnerOutline = "innerOutline";
export let matTextColor = 0xFFFFFF;
export let matWellColor = 0x006699;
let matWellSelectedColor = 0x0099FF; // Brighter blue for selected wells
export let matSiteColor = 0xFF8800;
export let matPlateColor = 0x222222;
export let matFovColor = 0x00CC00; // Strong green for objective/FOV
export let matForbiddenAreaColor = 0xCC4444; // Muted red for forbidden areas

/**
 * @typedef {Object} SelectionState
 * @property {boolean} active - Whether selection is currently active
 * @property {'select'|'deselect'} mode - Selection mode (select or deselect wells)
 * @property {{x: number, y: number}} start - Starting coordinates of selection
 * @property {{x: number, y: number}} current - Current coordinates of selection
 * @property {THREE.Mesh|null} box - THREE.Mesh for selection box visualization
 */

/**
 * @typedef {Object} MouseCoordinates
 * @property {number} offsetX - X coordinate relative to the element
 * @property {number} offsetY - Y coordinate relative to the element
 */

/**
 * @typedef {{minX:number,maxX:number,minY:number,maxY:number}} WellSelectionBounds
 */

export class PlateNavigator {
    /**
     *
     * @param {HTMLElement} containerel
     * @param {object} alpineComponent - Alpine.js component for updating reactive data
     * @returns
     */
    constructor(containerel, alpineComponent) {
        // Store reference to Alpine component for updating cursor position
        this.alpineComponent = alpineComponent;
        /** @type {THREE.Scene} */
        this.scene = new THREE.Scene();

        // setup scene
        let frame = {
            width: containerel.clientWidth,
            height: containerel.clientHeight,
        };
        /** @type {{left:number,right:number,top:number,bottom:number,near:number,far:number,zoom:number}} */
        this.cam = {
            left: frame.width / -2,
            right: frame.width / 2,
            top: frame.height / 2,
            bottom: frame.height / -2,
            near: -1,
            far: 1,

            zoom: 1,
        };
        // https://threejs.org/docs/#api/en/cameras/OrthographicCamera
        /** @type {THREE.OrthographicCamera} */
        this.camera = new THREE.OrthographicCamera(
            this.cam.left, this.cam.right,
            this.cam.top, this.cam.bottom,
            this.cam.near, this.cam.far
        );

        // try webgl and webgpu renderers, and accept whichever works
        /** @type {THREE.Renderer} */
        this.renderer;
        try { this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" }) } catch (e) { }
        //try { this.renderer = new THREE.WebGPURenderer({ antialias: true, powerPreference: "high-performance" }) } catch (e) { }
        if (!this.renderer) throw new Error(`no valid renderer found`);

        this.renderer.setSize(frame.width, frame.height);
        this.renderer.setPixelRatio(window.devicePixelRatio); // enable ssaa by *1.5 (bad for performance)
        /**@type {HTMLElement} */
        let el = this.renderer.domElement;
        containerel.appendChild(el);

        // threejs dragcontrols are only for dragging a 3d object, not for dragging the camera
        let drag = {
            active: false,
            last: { x: 0, y: 0 },
        };
        
        // Selection state for shift+drag well selection
        /** @type {SelectionState} */
        let selection = {
            active: false,
            mode: 'select', // 'select' for adding wells, 'deselect' for removing wells
            start: { x: 0, y: 0 },
            current: { x: 0, y: 0 },
            box: null, // THREE.Mesh for selection box visualization
        };

        el.addEventListener("mousedown", event => {
            if (event.shiftKey) {
                // Prevent default behavior and context menu
                event.preventDefault();
                event.stopPropagation();
                
                // Make sure drag is not active when starting selection
                drag.active = false;
                
                // Start selection mode
                selection.active = true;
                selection.mode = event.button === 2 ? 'deselect' : 'select';
                selection.start.x = event.offsetX;
                selection.start.y = event.offsetY;
                selection.current.x = event.offsetX;
                selection.current.y = event.offsetY;
                
                // Create selection box mesh
                this.createSelectionBox(selection);
                
            } else {
                // Start drag mode only if not in selection mode
                drag.active = true;
                drag.last.x = event.offsetX;
                drag.last.y = event.offsetY;
            }
        });
        el.addEventListener("mouseup", event => {
            if (selection.active) {
                // Prevent default behavior when ending selection
                event.preventDefault();
                event.stopPropagation();
                
                // End selection mode - perform well selection
                selection.active = false;
                this.performWellSelection(selection);
                this.clearSelectionBox(selection);
            }
            // Always reset drag state on mouse up
            drag.active = false;
        });
        
        // Add double-click handler for objective movement
        el.addEventListener("dblclick", event => {
            this.handleDoubleClick(event);
        });
        el.addEventListener("mousemove", event => {
            // Update cursor position display
            const plateCoords = this.mouseToPlateCoordinates(event);
            if (plateCoords && this.alpineComponent) {
                this.alpineComponent.plateCursorPosition = plateCoords;
            }

            if (selection.active) {
                // Prevent default behavior when in selection mode
                event.preventDefault();
                event.stopPropagation();

                // Update selection box
                selection.current.x = event.offsetX;
                selection.current.y = event.offsetY;
                this.updateSelectionBox(selection);
            } else if (drag.active) {
                // update camera view (based on current zoom factor)
                this.cameraApplyDeltaPos({
                    x: event.offsetX - drag.last.x,
                    y: event.offsetY - drag.last.y,
                });

                drag.last.x = event.offsetX;
                drag.last.y = event.offsetY;
            }

        });

        // Clear cursor position when mouse leaves the plate area
        el.addEventListener("mouseleave", event => {
            if (this.alpineComponent) {
                this.alpineComponent.plateCursorPosition = null;
            }
        });

        // Prevent context menu when shift+right-clicking
        el.addEventListener("contextmenu", event => {
            if (event.shiftKey) {
                event.preventDefault();
                event.stopPropagation();
            }
        });

        containerel.addEventListener("wheel", event => {
            event.preventDefault();

            const scroll_speed = 2e-3;

            // calculate zoom factor
            let delta_zoom = event.deltaY * scroll_speed;
            let xratio = event.offsetX / frame.width;
            let yratio = event.offsetY / frame.height;

            this.cameraApplyDeltaZoom(delta_zoom, {
                x: xratio,
                y: yratio,
            });
        }, { capture: true, passive: false });

        this.display_time = false;
        el.addEventListener("keydown", event => {
            if (event.key == "f") {
                this.display_time = true;
            }
        });

        /**
         * set up object and add to scene
         * @type {FontLoader}
         */
        const fontloader = new FontLoader();

        // font from three.js example
        /** @type {Promise<THREE.Font>} */
        this.font = new Promise((resolve, reject) => fontloader.load("resources/helvetiker_regular.typeface.json", font => {
            resolve(font);
        }));

        // create material to draw shapes with
        /** @type {THREE.LineBasicMaterial} */
        this.matText = new THREE.LineBasicMaterial({
            color: matTextColor,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.6,
        });
        /** @type {THREE.LineBasicMaterial} */
        this.matWell = new THREE.LineBasicMaterial({
            color: matWellColor,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.55,
        });
        /** @type {THREE.LineBasicMaterial} */
        this.matWellSelected = new THREE.LineBasicMaterial({
            color: matWellSelectedColor,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.6,
        });
        /** @type {THREE.LineBasicMaterial} */
        this.matSite = new THREE.LineBasicMaterial({
            color: matSiteColor,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.5,
        });
        /** @type {THREE.LineBasicMaterial} */
        this.matSiteUnselected = new THREE.LineBasicMaterial({
            color: 0x666666,  // Gray color for unselected sites instead of opacity
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.5,  // Same opacity as selected but different color
        });
        /** @type {THREE.LineBasicMaterial} */
        this.matPlate = new THREE.LineBasicMaterial({
            color: matPlateColor,
            side: THREE.DoubleSide,
        });
        /** @type {THREE.LineBasicMaterial} */
        this.matFov = new THREE.LineBasicMaterial({
            color: matFovColor,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.5,
        });
        /** @type {THREE.MeshBasicMaterial} */
        this.matForbiddenArea = new THREE.MeshBasicMaterial({
            color: matForbiddenAreaColor,
            side: THREE.DoubleSide,
        });

        /** @type {THREE.LineBasicMaterial} */
        this.matOutlineBlack = new THREE.LineBasicMaterial({
            color: 0xCCCCCC,
            linewidth: 6,
        });

        /** @type {{fovx:number,fovy:number}} */
        this.objective = {
            // valid for 20x objective on original squid
            fovx: 0.9,
            fovy: 0.9,
        };

        // Create group to wrap FOV mesh for centered scaling during pulse animation
        this.objectiveFovGroup = new THREE.Group();
        this.objectiveFovGroup.position.z = 0.6;
        this.scene.add(this.objectiveFovGroup);

        /** @type {THREE.Mesh} */
        this.objectiveFov = makeQuad({
            ax: 0, ay: 0,
            bx: this.objective.fovx, by: this.objective.fovy,
        }, this.matFov);
        // Position mesh within group so it's centered around group origin
        this.objectiveFov.position.x = -this.objective.fovx / 2;
        this.objectiveFov.position.y = -this.objective.fovy / 2;
        this.objectiveFovGroup.add(this.objectiveFov);

        // Animation state for objective FOV pulse
        this.fovPulseActive = false;
        this.fovPulseStartTime = 0;

        // setup render loop
        /** @type {number} */
        this.framenum = 0;

        /** @type {number} */
        this.last_frame = performance.now();

        this.renderer.setAnimationLoop(() => {
            this.animate();
        });

        /**@type {Wellplate|null} */
        this.plate = null;
    }

    /**
     * 
     * @param {Pos2} pos 
     */
    cameraApplyDeltaPos(pos) {
        this.camera.left -= pos.x * this.cam.zoom;
        this.camera.right -= pos.x * this.cam.zoom;
        this.camera.top += pos.y * this.cam.zoom;
        this.camera.bottom += pos.y * this.cam.zoom;
        this.camera.updateProjectionMatrix();
    }

    /**
     * 
     * @param {number} zoom
     * @returns 
     */
    cameraZoomAbsoluteToRelative(zoom) {
        return (zoom / this.cam.zoom) - 1;
    }

    /**
     * 
     * @param {number} delta 
     * @param {Pos2} [ratio={x:0.5,y:0.5}]
     */
    cameraApplyDeltaZoom(delta, ratio = { x: 0.5, y: 0.5 }) {
        // update absolute zoom factor
        this.cam.zoom *= 1 + delta;

        // zoom in on cursor position
        let deltax = this.camera.right - this.camera.left;
        let deltay = this.camera.top - this.camera.bottom;

        this.camera.left -= deltax * ratio.x * delta;
        this.camera.right += deltax * (1 - ratio.x) * delta;
        this.camera.top += deltay * ratio.y * delta;
        this.camera.bottom -= deltay * (1 - ratio.y) * delta;

        this.camera.updateProjectionMatrix();
    }
    /**
     * 
     * @param {Pos2} pos 
     */
    cameraSetCenter(pos) {
        let camera_center = {
            x: (this.camera.left + (this.camera.right - this.camera.left) / 2),
            y: (this.camera.bottom + (this.camera.top - this.camera.bottom) / 2),
        };
        let delta = {
            x: - (pos.x - camera_center.x),
            y: pos.y - camera_center.y,
        };
        this.cameraApplyDeltaPos(delta);
        camera_center = {
            x: (this.camera.left + (this.camera.right - this.camera.left) / 2),
            y: (this.camera.bottom + (this.camera.top - this.camera.bottom) / 2),
        };
    }

    /**
     * 
     * @param {AABB} aabb 
     */
    cameraFit(aabb) {
        let extentx = aabb.bx - aabb.ax;
        let centerx = aabb.ax + extentx / 2;
        let extenty = aabb.by - aabb.ay;
        let centery = aabb.ay + extenty / 2;

        this.cameraSetCenter({
            x: centerx,
            y: centery,
        });

        let targetzoomx = (extentx) / (this.camera.right - this.camera.left);
        let targetzoomy = extenty / (this.camera.top - this.camera.bottom);

        let targetzoom = Math.max(targetzoomx, targetzoomy);
        this.cameraApplyDeltaZoom(this.cameraZoomAbsoluteToRelative(targetzoom));
    }

    /**
     * 
     * @param {AcquisitionConfig} microscope_config
     * @param {Wellplate} plate
     */
    async loadPlate(microscope_config, plate) {
        // clear previous state
        this.clearState(plate);
        // reset camera zoom AND bounds together to maintain synchronization
        // (drag function uses this.cam.zoom to scale movement, but if bounds don't match the zoom level, drag will be wrong)
        const frame = {
            width: this.renderer.domElement.clientWidth,
            height: this.renderer.domElement.clientHeight,
        };
        this.camera.left = frame.width / -2;
        this.camera.right = frame.width / 2;
        this.camera.top = frame.height / 2;
        this.camera.bottom = frame.height / -2;
        this.camera.updateProjectionMatrix();
        this.cam.zoom = 1;
        // register plate as new state
        this.plate = plate;

        let plate_z = -0.1;
        let forbidden_area_z = -0.05;
        let well_z = 0;
        let site_z = 0.1;
        let welltext_z = 0.5;

        let plate_aabb = {
            ax: 0,
            ay: 0,
            bx: plate.Length_mm,
            by: plate.Width_mm,
        };
        let platemesh = makeQuad(plate_aabb, this.matPlate);
        platemesh.position.z = plate_z;
        platemesh.name = objectNamePlate;
        this.scene.add(platemesh);

        // Render forbidden areas
        try {
            const forbiddenAreas = await this.fetchForbiddenAreas();
            if (forbiddenAreas.length > 0) {
                const forbiddenGeometries = [];

                for (let i = 0; i < forbiddenAreas.length; i++) {
                    const area = forbiddenAreas[i];

                    // Validate area bounds
                    if (typeof area.min_x_mm !== 'number' || typeof area.max_x_mm !== 'number' ||
                        typeof area.min_y_mm !== 'number' || typeof area.max_y_mm !== 'number') {
                        console.warn(`Skipping invalid forbidden area: ${area.name || 'unnamed'}`);
                        continue;
                    }

                    // Transform backend coordinates to display coordinates
                    const minCoords = transformBackendToDisplayCoordinates(area.min_x_mm, area.min_y_mm, plate);
                    const maxCoords = transformBackendToDisplayCoordinates(area.max_x_mm, area.max_y_mm, plate);

                    const area_aabb = {
                        ax: minCoords.x,
                        ay: Math.min(minCoords.y, maxCoords.y), // Ensure min is actually minimum after transformation
                        bx: maxCoords.x,
                        by: Math.max(minCoords.y, maxCoords.y), // Ensure max is actually maximum after transformation
                    };

                    const forbiddenMesh = makeQuad(area_aabb, this.matForbiddenArea);
                    forbiddenGeometries.push(forbiddenMesh.geometry);
                }

                // Merge all forbidden area geometries into a single mesh to avoid transparency overlap issues
                if (forbiddenGeometries.length > 0) {
                    const mergedGeometry = BufferGeometryUtils.mergeGeometries(forbiddenGeometries);
                    const mergedMesh = new THREE.Mesh(mergedGeometry, this.matForbiddenArea);
                    mergedMesh.position.z = forbidden_area_z;
                    mergedMesh.name = objectNameForbiddenAreas;
                    this.scene.add(mergedMesh);
                }
            }
        } catch (error) {
            console.warn('Failed to render forbidden areas:', error);
        }

        // Remove old outlines before creating new ones
        const oldOuterOutline = this.scene.getObjectByName(objectNameOuterOutline);
        if (oldOuterOutline) this.scene.remove(oldOuterOutline);
        const oldInnerOutline = this.scene.getObjectByName(objectNameInnerOutline);
        if (oldInnerOutline) this.scene.remove(oldInnerOutline);

        // Draw plate outlines as thin rectangles so they scale with zoom
        const outline_z = 0.1;
        const stroke_width = 0.3; // mm - thickness of outline strokes

        // Outer plate boundary - draw as 4 thin rectangles
        const outerStrokeGeometries = [];

        // Top edge
        outerStrokeGeometries.push(makeQuad({
            ax: 0, ay: plate.Width_mm - stroke_width,
            bx: plate.Length_mm, by: plate.Width_mm
        }, this.matOutlineBlack).geometry);

        // Bottom edge
        outerStrokeGeometries.push(makeQuad({
            ax: 0, ay: 0,
            bx: plate.Length_mm, by: stroke_width
        }, this.matOutlineBlack).geometry);

        // Left edge
        outerStrokeGeometries.push(makeQuad({
            ax: 0, ay: 0,
            bx: stroke_width, by: plate.Width_mm
        }, this.matOutlineBlack).geometry);

        // Right edge
        outerStrokeGeometries.push(makeQuad({
            ax: plate.Length_mm - stroke_width, ay: 0,
            bx: plate.Length_mm, by: plate.Width_mm
        }, this.matOutlineBlack).geometry);

        const outerOutlineMerged = BufferGeometryUtils.mergeGeometries(outerStrokeGeometries);
        const outerOutline = new THREE.Mesh(outerOutlineMerged, this.matOutlineBlack);
        outerOutline.position.z = outline_z;
        outerOutline.name = objectNameOuterOutline;
        this.scene.add(outerOutline);

        // Inner well boundary with 45-degree chamfer at A1 corner (top-left)
        const inset_left = 2;
        const inset_right = 2;
        const inset_top = 2;
        const inset_bottom = 2;
        const chamfer_size = 4; // 4mm chamfer at A1 corner

        const innerX1 = inset_left;
        const innerX2 = plate.Length_mm - inset_right;
        const innerY1 = inset_bottom;
        const innerY2 = plate.Width_mm - inset_top;

        const innerStrokeGeometries = [];

        // Bottom edge
        innerStrokeGeometries.push(makeQuad({
            ax: innerX1, ay: innerY1,
            bx: innerX2, by: innerY1 + stroke_width
        }, this.matOutlineBlack).geometry);

        // Right edge
        innerStrokeGeometries.push(makeQuad({
            ax: innerX2 - stroke_width, ay: innerY1,
            bx: innerX2, by: innerY2
        }, this.matOutlineBlack).geometry);

        // Top-right to chamfer start
        innerStrokeGeometries.push(makeQuad({
            ax: innerX1 + chamfer_size, ay: innerY2 - stroke_width,
            bx: innerX2, by: innerY2
        }, this.matOutlineBlack).geometry);

        // Left edge (below chamfer)
        innerStrokeGeometries.push(makeQuad({
            ax: innerX1, ay: innerY1 + stroke_width,
            bx: innerX1 + stroke_width, by: innerY2 - chamfer_size
        }, this.matOutlineBlack).geometry);

        // Diagonal chamfer at A1 corner (45-degree cut)
        const chamferShape = new THREE.Shape();
        chamferShape.moveTo(innerX1, innerY2 - chamfer_size);
        chamferShape.lineTo(innerX1 + chamfer_size, innerY2);
        chamferShape.lineTo(innerX1 + chamfer_size + stroke_width * 0.7, innerY2 - stroke_width * 0.7);
        chamferShape.lineTo(innerX1 + stroke_width * 0.7, innerY2 - chamfer_size - stroke_width * 0.7);
        chamferShape.lineTo(innerX1, innerY2 - chamfer_size);
        const chamferGeo = new THREE.ShapeGeometry(chamferShape);
        innerStrokeGeometries.push(chamferGeo);

        const innerOutlineMerged = BufferGeometryUtils.mergeGeometries(innerStrokeGeometries);
        const innerOutline = new THREE.Mesh(innerOutlineMerged, this.matOutlineBlack);
        innerOutline.position.z = outline_z;
        innerOutline.name = objectNameInnerOutline;
        this.scene.add(innerOutline);

        // Fit camera to show the entire plate properly
        this.cameraFit(plate_aabb);

        // display wells with different colors for selected vs unselected
        let well_geometry = makeRoundedQuad({
            ax: 0,
            ay: 0,
            bx: 0 + plate.Well_size_x_mm,
            by: 0 + plate.Well_size_y_mm,
        }, this.matWell, { border_radius: plate.Well_edge_radius_mm, segments: 8 }).geometry;

        let valid_wells = microscope_config.plate_wells.filter(w => w.selected && w.col >= 0 && w.row >= 0);
        let unselected_wells = microscope_config.plate_wells.filter(w => !w.selected && w.col >= 0 && w.row >= 0);
        
        /**@type {THREE.InstancedMesh?} */
        let well_quads = null;
        /**@type {THREE.InstancedMesh?} */
        let well_selected_quads = null;
        
        if (!this.scene.getObjectByName(objectNameWells) && unselected_wells.length > 0) {
            well_quads = new THREE.InstancedMesh(well_geometry, this.matWell, unselected_wells.length);
        }
        if (!this.scene.getObjectByName(objectNameWellsSelected) && valid_wells.length > 0) {
            well_selected_quads = new THREE.InstancedMesh(well_geometry, this.matWellSelected, valid_wells.length);
        }

        let site_quad = makeQuad({
            ax: 0,
            ay: 0,
            bx: 0 + this.objective.fovx,
            by: 0 + this.objective.fovy,
        }, this.matSite);
        /**@type {THREE.InstancedMesh|null} */
        let site_quads_selected = null;
        let site_quads_unselected = null;
        let num_sites_selected = valid_wells.length * microscope_config.grid.mask.filter(s => s.selected).length;
        let num_sites_unselected = valid_wells.length * microscope_config.grid.mask.filter(s => !s.selected).length;
        
        if (!this.scene.getObjectByName(objectNameSites + "_selected")) {
            if (num_sites_selected > 0) {
                site_quads_selected = new THREE.InstancedMesh(
                    site_quad.geometry,
                    this.matSite,
                    num_sites_selected
                );
            }
        }
        
        if (!this.scene.getObjectByName(objectNameSites + "_unselected")) {
            if (num_sites_unselected > 0) {
                site_quads_unselected = new THREE.InstancedMesh(
                    site_quad.geometry,
                    this.matSiteUnselected,
                    num_sites_unselected
                );
            }
        }

        // combine all text into one mesh to simplify into a single draw call
        // (would be even better to instance the letters to save memory.. but still better than naive)
        /**@type {THREE.Geometry[]|null} */
        let text_geometries = null;
        if (!this.scene.getObjectByName(objectNameText)) {
            text_geometries = [];
        }

        let text_font = await this.font;

        // init index into site buffer (will be incremented before first use)
        let site_index_selected = -1;
        let site_index_unselected = -1;
        let selected_well_index = 0;
        let unselected_well_index = 0;
        
        // display all wells, separating selected and unselected
        for (let well of microscope_config.plate_wells) {
            let x = well.col;
            let y = well.row;

            // skip invalid wells
            if (x < 0 || y < 0) {
                continue
            }

            // add well
            const wellPos = calculateWellPosition(plate, x, y);
            let well_x = wellPos.x;
            let well_y = wellPos.y;

            let translatematrix = new THREE.Vector3(well_x, well_y, well_z);
            let quaternion = new THREE.Quaternion();
            quaternion.identity();
            let scalematrix = new THREE.Vector3(1, 1, 1);

            let well_matrix = new THREE.Matrix4();
            well_matrix.identity();
            well_matrix.compose(translatematrix, quaternion, scalematrix);

            // Use appropriate InstancedMesh based on selection state
            if (well.selected) {
                if (well_selected_quads != null) {
                    well_selected_quads.setMatrixAt(selected_well_index, well_matrix);
                    selected_well_index++;
                }
            } else {
                if (well_quads != null) {
                    well_quads.setMatrixAt(unselected_well_index, well_matrix);
                    unselected_well_index++;
                }
            }

            /** @type {AABB} */
            let well_aabb = {
                ax: translatematrix.x,
                ay: translatematrix.y,
                bx: translatematrix.x + plate.Well_size_x_mm,
                by: translatematrix.y + plate.Well_size_y_mm,
            };

            // add sites to well
            if (site_quads_selected != null || site_quads_unselected != null) {
                // add well to display, but skip sites, if well is not selected
                if (well.selected) {
                    for (let site of microscope_config.grid.mask) {
                        let sitex = site.col;
                        let sitey = site.row;

                        const sitePos = calculateSitePosition(
                            well_x, well_y, plate.Well_size_x_mm, plate.Well_size_y_mm,
                            sitex, sitey, microscope_config.grid.num_x, microscope_config.grid.num_y,
                            microscope_config.grid.delta_x_mm, microscope_config.grid.delta_y_mm,
                            this.objective.fovx, this.objective.fovy
                        );
                        let site_plate_x_offset = sitePos.x;
                        let site_plate_y_offset = sitePos.y;

                        let translatematrix = new THREE.Vector3(
                            site_plate_x_offset,
                            site_plate_y_offset,
                            site_z
                        );

                        let quaternion = new THREE.Quaternion();
                        quaternion.identity();
                        let scalematrix = new THREE.Vector3(1, 1, 1);

                        let site_matrix = new THREE.Matrix4();
                        site_matrix.identity();
                        site_matrix.compose(translatematrix, quaternion, scalematrix);

                        // Use appropriate mesh based on site selection state
                        if (site.selected && site_quads_selected != null) {
                            site_index_selected++;
                            site_quads_selected.setMatrixAt(site_index_selected, site_matrix);
                        } else if (!site.selected && site_quads_unselected != null) {
                            site_index_unselected++;
                            site_quads_unselected.setMatrixAt(site_index_unselected, site_matrix);
                        }
                    }
                }
            }

            if (text_geometries != null) {
                // generate text geometry from https://github.com/mrdoob/three.js/blob/master/examples/webgl_geometry_text_shapes.html
                // also see https://threejs.org/docs/index.html#manual/en/introduction/Creating-text
                let textstr = makeWellName(x, y);
                let fontsize_px = Math.min(plate.Well_size_x_mm, plate.Well_size_y_mm) / 5;

                let shapes = text_font.generateShapes(textstr, fontsize_px);
                let geometry = new THREE.ShapeGeometry(shapes);

                geometry.computeBoundingBox();
                const xMid = - 0.5 * (geometry.boundingBox.max.x - geometry.boundingBox.min.x);
                const yMid = - 0.5 * (geometry.boundingBox.max.y - geometry.boundingBox.min.y);

                // center text around its target position
                geometry.translate(
                    xMid + plate.Well_size_x_mm / 2 + well_aabb.ax,
                    yMid + plate.Well_size_y_mm / 2 + well_aabb.ay,
                    welltext_z
                );
                // add to text geometry
                text_geometries.push(geometry);
            }
        }
        // from https://github.com/mrdoob/three.js/blob/master/examples/webgl_instancing_performance.html
        if (text_geometries != null) {
            let text = BufferGeometryUtils.mergeGeometries(text_geometries);
            let textmesh = new THREE.Mesh(text, this.matText);
            textmesh.name = objectNameText;
            this.scene.add(textmesh);
        }

        if (well_quads != null) {
            well_quads.name = objectNameWells;
            this.scene.add(well_quads);
        }
        if (well_selected_quads != null) {
            well_selected_quads.name = objectNameWellsSelected;
            this.scene.add(well_selected_quads);
        }
        if (site_quads_selected != null) {
            site_quads_selected.name = objectNameSites + "_selected";
            if (this.scene.getObjectByName(objectNameSites + "_selected")) {
                console.error("duplicate found for", objectNameSites + "_selected");
            } else {
                this.scene.add(site_quads_selected);
            }
        }
        
        if (site_quads_unselected != null) {
            site_quads_unselected.name = objectNameSites + "_unselected";
            if (this.scene.getObjectByName(objectNameSites + "_unselected")) {
                console.error("duplicate found for", objectNameSites + "_unselected");
            } else {
                this.scene.add(site_quads_unselected);
            }
        }
    }

    animate() {
        this.framenum++;

        // Handle FOV pulse animation
        if (this.fovPulseActive && this.objectiveFovGroup) {
            const elapsed = performance.now() - this.fovPulseStartTime;
            const duration = 2500; // 2.5 seconds in milliseconds

            if (elapsed >= duration) {
                // Animation complete, reset scale to normal
                this.objectiveFovGroup.scale.set(1, 1, 1);
                this.fovPulseActive = false;
            } else {
                // Calculate progress (0 to 1)
                const progress = elapsed / duration;
                // Ease-out cubic: starts fast, ends slow for smooth deceleration
                const eased = 1 - Math.pow(1 - progress, 3);
                // Interpolate scale from 15 to 1
                const scale = 15 - (14 * eased);
                this.objectiveFovGroup.scale.set(scale, scale, 1);
            }
        }

        this.renderer.render(this.scene, this.camera);

        let delta = performance.now() - this.last_frame;
        this.last_frame = performance.now();
        if (this.display_time) {
            this.display_time = false;

            alert("frametime " + delta + " ms");
        }
    }

    /**
     * remove wells, well text and sites (if plate has changed)
     * @param {Wellplate?} [plate=null]
     */
    clearState(plate) {
        /** @type {string[]} */
        let objectNamesToRemove = [
            objectNameSites,
        ];

        if (this.plate != null && plate != null && this.plate?.Model_id == plate.Model_id) {
            // do nothing
            // (if same plate is used, assume only site settings have changed, so no need to regenerate everything else)
        } else {
            objectNamesToRemove.push(...[
                // plate likely unchanged, but very cheap to regenerate
                objectNamePlate,
            ]);

            // even if a plate has the same number of wells, the position or size of the wells
            // may be different (same for the text positions), hence it needs to be regenerated
            objectNamesToRemove.push(...[
                objectNameWells,
                objectNameWellsSelected,
                objectNameText,
                objectNameOuterOutline,
                objectNameInnerOutline,
            ]);
        }

        // Always remove forbidden areas since they depend on machine configuration
        // and may change independently of plate configuration
        for (let i = 0; i < 100; i++) { // Remove up to 100 forbidden areas (generous limit)
            const forbiddenAreaName = `${objectNameForbiddenAreas}_${i}`;
            const forbiddenObject = this.scene.getObjectByName(forbiddenAreaName);
            if (forbiddenObject) {
                this.scene.remove(forbiddenObject);
            } else if (i > 10) {
                // If we haven't found any objects for 10 consecutive attempts, stop looking
                break;
            }
        }

        for (const objectname of objectNamesToRemove) {
            // from https://stackoverflow.com/questions/18357529/threejs-remove-object-from-scene
            let object_to_remove = this.scene.getObjectByName(objectname);
            if (object_to_remove) {
                this.scene.remove(object_to_remove);
            }
        }
    }

    /**
     * Handle double-click events to move objective to clicked position
     * @param {MouseEvent} event
     */
    async handleDoubleClick(event) {
        if (!this.plate) {
            console.warn("Cannot move objective - no plate loaded");
            return;
        }

        // Convert mouse coordinates to plate coordinates
        const plateCoords = this.mouseToPlateCoordinates(event);

        if (plateCoords && this.onObjectiveMoveTo) {
            // Validate coordinates before sending to server
            if (this.isPositionValid(plateCoords.x, plateCoords.y)) {
                await this.onObjectiveMoveTo(plateCoords.x, plateCoords.y);
            } else {
                // Show user-friendly error message
                this.showForbiddenAreaWarning(plateCoords.x, plateCoords.y);
            }
        }
    }

    /**
     * Basic client-side validation for movement positions
     * Note: This is a preliminary check - the server will do the authoritative validation
     * @param {number} x_mm - X coordinate in mm
     * @param {number} y_mm - Y coordinate in mm
     * @returns {boolean} True if position appears valid for movement
     */
    isPositionValid(x_mm, y_mm) {
        if (!this.plate) return false;

        // Check if position is within plate boundaries
        if (x_mm < 0 || x_mm > this.plate.Length_mm ||
            y_mm < 0 || y_mm > this.plate.Width_mm) {
            return false;
        }

        // Basic sanity checks passed - server will do hardware validation
        return true;
    }

    /**
     * Show warning message when user tries to move to a forbidden area
     * @param {number} x_mm - X coordinate in mm
     * @param {number} y_mm - Y coordinate in mm
     */
    showForbiddenAreaWarning(x_mm, y_mm) {
        // Create a simple alert for now - could be replaced with a more sophisticated UI
        const message = `Cannot move to position (${x_mm.toFixed(1)}, ${y_mm.toFixed(1)}) mm.\n` +
                       `This position is outside the valid movement area.`;
        alert(message);

        console.warn(`Blocked movement to potentially forbidden position: (${x_mm}, ${y_mm})`);
    }

    /**
     * Convert mouse coordinates to plate coordinates (in mm)
     * Returns backend/physical coordinates with origin at top-left (A1 position)
     * @param {MouseCoordinates} event - Object with offsetX and offsetY properties
     * @returns {{x: number, y: number} | null} Backend/physical plate coordinates in mm
     */
    mouseToPlateCoordinates(event) {
        if (!this.plate) return null;

        // Get canvas dimensions
        const canvasRect = this.renderer.domElement.getBoundingClientRect();

        // Convert mouse position to normalized coordinates (0 to 1)
        const normalizedX = event.offsetX / canvasRect.width;
        const normalizedY = event.offsetY / canvasRect.height;

        // Convert to camera space coordinates
        const cameraWidth = this.camera.right - this.camera.left;
        const cameraHeight = this.camera.top - this.camera.bottom;

        const cameraX = this.camera.left + normalizedX * cameraWidth;
        const cameraY = this.camera.bottom + (1 - normalizedY) * cameraHeight; // Flip Y axis

        // Clamp coordinates to plate boundaries (in display/camera space)
        const displayX = Math.max(0, Math.min(this.plate.Length_mm, cameraX));
        const displayY = Math.max(0, Math.min(this.plate.Width_mm, cameraY));

        // Transform from display coordinates (origin bottom) to backend/physical coordinates (origin top)
        const backendY = this.plate.Width_mm - displayY;

        return { x: displayX, y: backendY };
    }

    /**
     * Set callback function for objective movement requests
     * @param {function(number, number): void} callback - Callback function that receives (x_mm, y_mm)
     */
    setObjectiveMoveCallback(callback) {
        this.onObjectiveMoveTo = callback;
    }

    /**
     * Set callback function for handling movement errors (e.g., forbidden areas)
     * @param {function(string): void} callback - Callback function that receives error message
     */
    setMovementErrorCallback(callback) {
        this.onMovementError = callback;
    }

    /**
     * Handle movement errors from the server (e.g., forbidden area violations)
     * @param {string} errorMessage - Error message from server
     */
    handleMovementError(errorMessage) {
        if (this.onMovementError) {
            this.onMovementError(errorMessage);
        } else {
            // Fallback to alert if no custom error handler is set
            alert(`Movement Error: ${errorMessage}`);
        }
        console.error(`Server rejected movement: ${errorMessage}`);
    }

    /**
     * Set callback function for well selection requests
     * @param {function(string[], string, WellSelectionBounds): void} callback - Callback function that receives array of well names and mode ('select' or 'deselect')
     */
    setWellSelectionCallback(callback) {
        this.onWellSelection = callback;
    }

    /**
     * Fetch forbidden areas configuration from server
     * @returns {Promise<Array<object>>} Array of forbidden area objects
     */
    async fetchForbiddenAreas() {
        try {
            // Use the API client from the parent (Alpine component)
            const configItems = await this.alpineComponent.api.post('/api/get_features/machine_defaults', {}, {
                context: 'Fetch machine defaults for forbidden areas',
                showError: false
            });

            // Find the forbidden areas config item
            const forbiddenAreasItem = configItems.find(item => item.handle === 'protocol.forbidden_areas');

            if (!forbiddenAreasItem || !forbiddenAreasItem.value) {
                console.log('No forbidden areas configuration found');
                return [];
            }

            // Parse the JSON configuration (array of forbidden area objects)
            const forbiddenAreasConfig = JSON.parse(forbiddenAreasItem.value);
            return forbiddenAreasConfig || [];

        } catch (error) {
            console.warn('Error fetching forbidden areas:', error);
            return [];
        }
    }

    /**
     * Create selection box visualization
     * @param {SelectionState} selection - Selection state object
     */
    createSelectionBox(selection) {
        const color = selection.mode === 'select' ? 0x0088ff : 0xff4444;
        
        const material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });

        const geometry = new THREE.PlaneGeometry(1, 1);
        selection.box = new THREE.Mesh(geometry, material);
        selection.box.position.z = 0.8;
        selection.box.scale.set(0.1, 0.1, 1);
        this.scene.add(selection.box);
        
    }

    /**
     * Update selection box geometry and position
     * @param {SelectionState} selection - Selection state object
     */
    updateSelectionBox(selection) {
        if (!selection.box) return;

        // Convert screen coordinates to plate coordinates (backend coordinate system)
        const startBackend = this.mouseToPlateCoordinates({offsetX: selection.start.x, offsetY: selection.start.y});
        const currentBackend = this.mouseToPlateCoordinates({offsetX: selection.current.x, offsetY: selection.current.y});

        if (!startBackend || !currentBackend) return;

        // Transform from backend to display coordinates for Three.js rendering
        if(!this.plate)throw new Error(`plate is null`);
        const startDisplay = transformBackendToDisplayCoordinates(startBackend.x, startBackend.y, this.plate);
        const currentDisplay = transformBackendToDisplayCoordinates(currentBackend.x, currentBackend.y, this.plate);

        // Calculate box dimensions and position in display coordinate system
        const minX = Math.min(startDisplay.x, currentDisplay.x);
        const maxX = Math.max(startDisplay.x, currentDisplay.x);
        const minY = Math.min(startDisplay.y, currentDisplay.y);
        const maxY = Math.max(startDisplay.y, currentDisplay.y);

        const width = Math.max(maxX - minX, 0.1);
        const height = Math.max(maxY - minY, 0.1);
        const centerX = minX + (maxX - minX) / 2;
        const centerY = minY + (maxY - minY) / 2;

        selection.box.scale.set(width, height, 1);
        selection.box.position.set(centerX, centerY, 0.8);
    }

    /**
     * Remove selection box from scene
     * @param {SelectionState} selection - Selection state object
     */
    clearSelectionBox(selection) {
        if (selection.box) {
            this.scene.remove(selection.box);
            selection.box.geometry.dispose();
            selection.box.material.dispose();
            selection.box = null;
        }
    }

    /**
     * Update the objective position indicator based on current stage position
     * @param {number} x_mm - X coordinate in mm (backend coordinates)
     * @param {number} y_mm - Y coordinate in mm (backend coordinates)
     */
    updateObjectivePosition(x_mm, y_mm) {
        if (!this.plate || !this.objectiveFovGroup) {
            console.warn('Position update skipped - no plate or objectiveFovGroup');
            return; // No plate loaded or objective FOV not initialized
        }

        // Transform backend coordinates to display coordinates
        const displayCoords = transformBackendToDisplayCoordinates(x_mm, y_mm, this.plate);

        // Position group at center of FOV (mesh within group is offset by -fov/2 to be centered)
        this.objectiveFovGroup.position.x = displayCoords.x;
        this.objectiveFovGroup.position.y = displayCoords.y;

        // Keep the Z position as set during initialization (above other elements)
        // this.objectiveFovGroup.position.z is already set to 0.6 during creation
    }

    /**
     * Trigger a pulse animation for the objective FOV to highlight it
     * Scales from 15x to normal size over 2.5 seconds around the center of the FOV
     */
    pulseObjectiveFov() {
        this.fovPulseActive = true;
        this.fovPulseStartTime = performance.now();
    }

    /**
     * Refresh well colors and sites after selection changes without full plate reload
     * @param {AcquisitionConfig} microscope_config
     */
    refreshWellColors(microscope_config) {
        if (!this.plate) return;

        // Store the microscope config for cursor debugging
        this.microscopeConfig = microscope_config;

        // Remove existing well and site meshes
        const existingWells = this.scene.getObjectByName(objectNameWells);
        const existingSelectedWells = this.scene.getObjectByName(objectNameWellsSelected);
        const existingSitesSelected = this.scene.getObjectByName(objectNameSites + "_selected");
        const existingSitesUnselected = this.scene.getObjectByName(objectNameSites + "_unselected");
        // Also remove old single site mesh for backwards compatibility
        const existingSitesOld = this.scene.getObjectByName(objectNameSites);
        
        if (existingWells) {
            this.scene.remove(existingWells);
            existingWells.dispose();
        }
        if (existingSelectedWells) {
            this.scene.remove(existingSelectedWells);
            existingSelectedWells.dispose();
        }
        if (existingSitesSelected) {
            this.scene.remove(existingSitesSelected);
            existingSitesSelected.dispose();
        }
        if (existingSitesUnselected) {
            this.scene.remove(existingSitesUnselected);
            existingSitesUnselected.dispose();
        }
        if (existingSitesOld) {
            this.scene.remove(existingSitesOld);
            existingSitesOld.dispose();
        }

        // Recreate well meshes with current selections
        const plate = this.plate;
        let well_z = 0;
        
        let well_geometry = makeRoundedQuad({
            ax: 0,
            ay: 0,
            bx: 0 + plate.Well_size_x_mm,
            by: 0 + plate.Well_size_y_mm,
        }, this.matWell, { border_radius: plate.Well_edge_radius_mm, segments: 8 }).geometry;

        let valid_wells = microscope_config.plate_wells.filter(w => w.selected && w.col >= 0 && w.row >= 0);
        let unselected_wells = microscope_config.plate_wells.filter(w => !w.selected && w.col >= 0 && w.row >= 0);
        
        let well_quads = null;
        let well_selected_quads = null;
        
        if (unselected_wells.length > 0) {
            well_quads = new THREE.InstancedMesh(well_geometry, this.matWell, unselected_wells.length);
        }
        if (valid_wells.length > 0) {
            well_selected_quads = new THREE.InstancedMesh(well_geometry, this.matWellSelected, valid_wells.length);
        }

        let selected_well_index = 0;
        let unselected_well_index = 0;
        
        // Position wells
        for (let well of microscope_config.plate_wells) {
            let x = well.col;
            let y = well.row;

            if (x < 0 || y < 0) continue;

            const wellPos = calculateWellPosition(plate, x, y);
            let well_x = wellPos.x;
            let well_y = wellPos.y;

            let translatematrix = new THREE.Vector3(well_x, well_y, well_z);
            let quaternion = new THREE.Quaternion();
            quaternion.identity();
            let scalematrix = new THREE.Vector3(1, 1, 1);

            let well_matrix = new THREE.Matrix4();
            well_matrix.identity();
            well_matrix.compose(translatematrix, quaternion, scalematrix);

            if (well.selected) {
                if (well_selected_quads != null) {
                    well_selected_quads.setMatrixAt(selected_well_index, well_matrix);
                    selected_well_index++;
                }
            } else {
                if (well_quads != null) {
                    well_quads.setMatrixAt(unselected_well_index, well_matrix);
                    unselected_well_index++;
                }
            }
        }

        // Add wells to scene
        if (well_quads != null) {
            well_quads.name = objectNameWells;
            this.scene.add(well_quads);
        }
        if (well_selected_quads != null) {
            well_selected_quads.name = objectNameWellsSelected;
            this.scene.add(well_selected_quads);
        }

        // Recreate sites for selected wells
        let site_z = 0.1;
        let site_quad = makeQuad({
            ax: 0,
            ay: 0,
            bx: 0 + this.objective.fovx,
            by: 0 + this.objective.fovy,
        }, this.matSite);
        
        // Only count sites in selected wells
        let selected_wells = microscope_config.plate_wells.filter(w => w.selected && w.row >= 0 && w.col >= 0);
        let num_sites_selected = selected_wells.length * microscope_config.grid.mask.filter(s => s.selected).length;
        let num_sites_unselected = selected_wells.length * microscope_config.grid.mask.filter(s => !s.selected).length;
        let site_quads_selected = null;
        let site_quads_unselected = null;
        
        if (num_sites_selected > 0) {
            site_quads_selected = new THREE.InstancedMesh(
                site_quad.geometry,
                this.matSite,
                num_sites_selected
            );
        }
        
        if (num_sites_unselected > 0) {
            site_quads_unselected = new THREE.InstancedMesh(
                site_quad.geometry,
                this.matSiteUnselected,
                num_sites_unselected
            );
        }

        let site_index_selected = -1;
        let site_index_unselected = -1;
            
        // Add sites to selected wells
        for (let well of microscope_config.plate_wells) {
            let x = well.col;
            let y = well.row;

            if (x < 0 || y < 0 || !well.selected) continue;

            const wellPos = calculateWellPosition(plate, x, y);
            let well_x = wellPos.x;
            let well_y = wellPos.y;

            for (let site of microscope_config.grid.mask) {
                let sitex = site.col;
                let sitey = site.row;

                const sitePos = calculateSitePosition(
                    well_x, well_y, plate.Well_size_x_mm, plate.Well_size_y_mm,
                    sitex, sitey, microscope_config.grid.num_x, microscope_config.grid.num_y,
                    microscope_config.grid.delta_x_mm, microscope_config.grid.delta_y_mm,
                    this.objective.fovx, this.objective.fovy
                );
                let site_plate_x_offset = sitePos.x;
                let site_plate_y_offset = sitePos.y;

                let translatematrix = new THREE.Vector3(
                    site_plate_x_offset,
                    site_plate_y_offset,
                    site_z
                );

                let quaternion = new THREE.Quaternion();
                quaternion.identity();
                let scalematrix = new THREE.Vector3(1, 1, 1);

                let site_matrix = new THREE.Matrix4();
                site_matrix.identity();
                site_matrix.compose(translatematrix, quaternion, scalematrix);

                // Use appropriate mesh based on site selection state
                if (site.selected && site_quads_selected != null) {
                    site_index_selected++;
                    site_quads_selected.setMatrixAt(site_index_selected, site_matrix);
                } else if (!site.selected && site_quads_unselected != null) {
                    site_index_unselected++;
                    site_quads_unselected.setMatrixAt(site_index_unselected, site_matrix);
                }
            }
        }

        if (site_quads_selected != null) {
            site_quads_selected.name = objectNameSites + "_selected";
            this.scene.add(site_quads_selected);
        }
        
        if (site_quads_unselected != null) {
            site_quads_unselected.name = objectNameSites + "_unselected";
            this.scene.add(site_quads_unselected);
        }
    }

    /**
     * Perform well selection based on selection box
     * @param {SelectionState} selection - Selection state object
     */
    performWellSelection(selection) {
        if (!this.plate || !this.onWellSelection) return;

        // Convert screen coordinates to plate coordinates (backend coordinate system)
        const startBackend = this.mouseToPlateCoordinates({offsetX: selection.start.x, offsetY: selection.start.y});
        const currentBackend = this.mouseToPlateCoordinates({offsetX: selection.current.x, offsetY: selection.current.y});

        if (!startBackend || !currentBackend) return;

        // Calculate selection bounding box in backend coordinates
        /** @type {WellSelectionBounds} */
        const selectionBoxBackend = {
            minX: Math.min(startBackend.x, currentBackend.x),
            maxX: Math.max(startBackend.x, currentBackend.x),
            minY: Math.min(startBackend.y, currentBackend.y),
            maxY: Math.max(startBackend.y, currentBackend.y)
        };

        // Transform selection bounds from backend to display coordinates for well intersection testing
        const minDisplay = transformBackendToDisplayCoordinates(selectionBoxBackend.minX, selectionBoxBackend.minY, this.plate);
        const maxDisplay = transformBackendToDisplayCoordinates(selectionBoxBackend.maxX, selectionBoxBackend.maxY, this.plate);

        // Build selection bounds in display coordinates (accounting for potential Y-flip)
        const selectionBoxDisplay = {
            minX: Math.min(minDisplay.x, maxDisplay.x),
            maxX: Math.max(minDisplay.x, maxDisplay.x),
            minY: Math.min(minDisplay.y, maxDisplay.y),
            maxY: Math.max(minDisplay.y, maxDisplay.y)
        };

        // Find wells that intersect with selection box
        const selectedWells = [];

        for (let y = 0; y < this.plate.Num_wells_y; y++) {
            for (let x = 0; x < this.plate.Num_wells_x; x++) {
                // Calculate well position and bounds in display coordinates
                const wellPos = calculateWellPosition(this.plate, x, y);
                const wellX = wellPos.x;
                const wellY = wellPos.y;

                const wellBounds = {
                    minX: wellX,
                    maxX: wellX + this.plate.Well_size_x_mm,
                    minY: wellY,
                    maxY: wellY + this.plate.Well_size_y_mm
                };

                // AABB intersection test in display coordinate system
                const intersects = !(
                    wellBounds.maxX < selectionBoxDisplay.minX ||
                    wellBounds.minX > selectionBoxDisplay.maxX ||
                    wellBounds.maxY < selectionBoxDisplay.minY ||
                    wellBounds.minY > selectionBoxDisplay.maxY
                );

                if (intersects) {
                    const wellName = makeWellName(x, y);
                    selectedWells.push(wellName);
                }
            }
        }

        // Pass selection bounds in backend coordinates for site selection
        // (selectSitesByWellArea expects backend coordinates)
        /** @type {WellSelectionBounds} */
        const selectionBounds = {
            minX: selectionBoxBackend.minX,
            maxX: selectionBoxBackend.maxX,
            minY: selectionBoxBackend.minY,
            maxY: selectionBoxBackend.maxY
        };

        this.onWellSelection(selectedWells, selection.mode, selectionBounds);
    }

}
