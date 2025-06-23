declare module "alpine" {
    class Alpine {
        /** init alpine */
        static start(): void;
        /** register component */
        static data<T>(name:string,gen:(()=>(object&T))):void;
    }
}