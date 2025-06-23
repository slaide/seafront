import { Font } from "three";

declare module "three/addons/loaders/FontLoader" {
    class FontLoader {
        constructor();

        load(filename: string, onload: (font: Font) => void): void;
    }
}

export { }