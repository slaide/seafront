declare module "three" {
    /** [doc](https://threejs.org/docs/#api/en/math/Color) */
    class Color {
        constructor();
        constructor(cssColor:string);
        constructor(hexCode:number);
        setRGB(r: number, g: number, b: number): Color;
        setHex(hex: number): Color;
        set(value: number | string | Color): this;
    }

    class Renderer {
        /**
         * set canvas size
         * 
         * @param updateStyle if false: prevents any style changes to the output canvas.
         */
        setSize(width: number, height: number, updateStyle?: boolean): void;
        /**
         * Sets device pixel ratio. This is usually used for HiDPI device to prevent blurring output canvas.
         * 
         * (backing canvas size = real canvas size * ratio )
         */
        setPixelRatio(ratio: number): void;
        /**
         * Sets the clear color and opacity.
         * 
         * @param color 
         * @param alpha 
         */
        setClearColor(color: Color, alpha?: number): void;

        get domElement(): HTMLElement;

        /**
         * 
         * @param callback The function will be called every available frame. If null is passed it will stop any already ongoing animation.
         */
        setAnimationLoop(callback: (() => void) | undefined): void;

        render(object: Object3D, camera: Camera): void;

        setScissor(left: number, positiveYUpBottom: number, width: number, height: number): void;
        setViewport(left: number, positiveYUpBottom: number, width: number, height: number): void;

        setScissorTest(enabled: boolean): void;

        /**
         * Tells the renderer to clear its color, depth or stencil drawing buffer(s).
         * This method initializes the color buffer to the current clear color value.
         * @param color clear color buffer
         * @param depth clear depth buffer
         * @param stencil clear stencil buffer
         */
        clear(color?: boolean, depth?: boolean, stencil?: boolean): void;

        /**
         * Clear the color buffer. Equivalent to calling .clear( true, false, false ).
         */
        clearColor(): void;

        /**
         * Clear the depth buffer. Equivalent to calling .clear( false, true, false ).
         */
        clearDepth(): void;

        /**
         * Clear the stencil buffers. Equivalent to calling .clear( false, false, true).
         */
        clearStencil(): void;
    }
    class WebGLRenderer extends Renderer {
        /**
         * 
         * opts.canvas: if none is provided, will create one (which must then be manually added to the dom for display)
         * 
         * @param opts
         */
        constructor(opts: {
            canvas?: HTMLCanvasElement,
            antialias?: boolean,
            alpha?: boolean,
            powerPreference?: "high-performance"
        });
    }

    /** indicate double sided geometry as material option */
    class OneSide { }
    class DoubleSide { }
    type SidedNess = OneSide | DoubleSide

    class UnsignedByteType { }
    class UnsignedShortType { }
    type DataType = UnsignedByteType | UnsignedShortType;

    class Material {
        color: Color | number;
        dispose(): void;
    }
    class MeshBasicMaterial extends Material {
        constructor(opts: {
            color: Color | number,
            side: SidedNess,
            transparent?: boolean,
            wireframe?: boolean,
            opacity?: number,
        });
    }
    class LineBasicMaterial extends Material {
        constructor(opts: {
            color: Color | number,
            side?: SidedNess,
            transparent?: boolean,
            opacity?: number,
            linewidth?: number,
        });
    }

    type BoundingBox = { max: Vector2, min: Vector2 }
    class Geometry {
        translate(x: number, y: number, z: number): void;

        computeBoundingBox(): void;
        get boundingBox(): BoundingBox;
        dispose(): void;
    }

    class Object3D {
        /**
         * add another object as child of this object
         * @param object
         */
        add(object: Object3D): void;
        remove(object: Object3D): this;
        getObjectByName(name: string): Object3D | undefined;

        /** internal name for this mesh. can be used to retrieve this mesh from a scene. */
        name: string | undefined;

        parent: Object3D | undefined;
        position: Vector3;
        quaternion: Quaternion;
        scale: Vector3;

        get up(): Vector3;

        /** indicates if object should be drawn */
        visible: boolean;
        
        /** dispose of this object and free memory */
        dispose(): void;
    }

    class InstancedMesh extends Object3D {
        /**
         * 
         * @param geo instanced geometry
         * @param mat material for each instance
         * @param n max number of instances
         */
        constructor(geo: Geometry, mat: Material, n: number);

        setMatrixAt(index: number, mat: Matrix4): void;
    }

    class Mesh extends Object3D {
        constructor(geometry: Geometry, material: Material);

        get geometry(): Geometry;
        get material(): Material;
    }

    class Shape {
        moveTo(x: number, y: number): void;
        lineTo(x: number, y: number): void;
    }
    class ShapeGeometry extends Geometry {
        constructor(shape: Shape);
    }
    class BufferAttribute {
        constructor(data: Float32Array | Uint32Array, items_per_element: number);
    }
    class BufferGeometry extends Geometry {
        constructor();
        setIndex(data: BufferAttribute): void;
        setAttribute(name: string, data: BufferAttribute): void;
        setAttribute(name: string, data: BufferAttribute): void;
    }

    class Scene extends Object3D {
        constructor();
    }

    class RedIntegerFormat { }

    class NearestFilter { }
    class LinearFilter { }

    type TextureFilter = NearestFilter | LinearFilter;

    class Texture {
        minFilter: TextureFilter;
        magFilter: TextureFilter;
        type: DataType;

        set needsUpdate(flag: boolean);
        dispose(): void;
    }

    class ShaderMaterial extends Material {
        constructor(args: {
            uniforms?: {
                uTexture?: { value: Texture }
            },
            vertexShader: string,
            fragmentShader: string,
        });
    }

    // 5. Create a simple quad (a PlaneGeometry) on which the texture is drawn.
    class PlaneGeometry extends Geometry {
        constructor(width: number, height: number);
    }

    class DataTexture extends Texture {
        constructor(data: Uint16Array | Uint8Array, width: number, height: number, format: RedIntegerFormat, datatype: DataType);

        image: { data: Uint16Array | Uint8Array, width: number, height: number };
    }

    class Camera extends Object3D {
        updateProjectionMatrix(): void;
    }
    class OrthographicCamera extends Camera {
        constructor(
            left: number, right: number,
            top: number, bottom: number,
            near: number, far: number
        );

        left: number;
        right: number;
        top: number;
        bottom: number;
        near: number;
        far: number;

        /** aspect ratio */
        aspect: number;
    }

    /** not a real type! */
    class Matrix {
        /** turn itself into an identity matrix */
        identity(): void;
    }
    class Vector2 extends Matrix {
        constructor(x: number, y: number);

        x: number;
        y: number;
    }
    class Vector3 extends Matrix {
        constructor(x: number, y: number, z: number);

        x: number;
        y: number;
        z: number;
        
        set(x: number, y: number, z: number): this;
    }
    class Matrix4 extends Matrix {
        constructor();

        /** compose model matrix in place */
        compose(translate: Vector3, rotate: Quaternion, scale: Vector3): void;
    }
    class Quaternion extends Matrix {
        constructor();
    }

    class Font {
        generateShapes(text: string, fontsize_px: number): Shape;
    }
}

// this line ensures that the 'declare global' are visible by the LSP in other .js files
export { }