declare module "json5" {
    interface ParseOptions {
        reviver?: (key: string, value: any) => any;
    }

    interface StringifyOptions {
        replacer?: ((key: string, value: any) => any) | (string | number)[] | null;
        space?: string | number;
        quote?: string;
    }

    class JSON5 {
        static parse(text: string, reviver?: (key: string, value: any) => any): any;
        static parse(text: string, options?: ParseOptions): any;
        
        static stringify(value: any): string;
        static stringify(value: any, replacer?: ((key: string, value: any) => any) | (string | number)[] | null, space?: string | number): string;
        static stringify(value: any, options?: StringifyOptions): string;
    }

    export = JSON5;
}
