declare module "three/addons/capabilities/WebGL" {
    class WebGL {
        static isWebGL2Available(): boolean;
        static getWebGL2ErrorMessage(): HTMLElement;
    }
}
export default WebGL;
