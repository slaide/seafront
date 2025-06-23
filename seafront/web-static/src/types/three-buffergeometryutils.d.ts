import { Geometry } from "three";
declare module "three/addons/utils/BufferGeometryUtils" {
    function mergeGeometries(geometries: Geometry[]): Geometry;
}

export { }