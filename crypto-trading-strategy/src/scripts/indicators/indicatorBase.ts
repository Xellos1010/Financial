// src/scripts/indicators/indicatorBase.ts
export abstract class IndicatorBase {
    protected prices: number[];

    constructor(prices: number[]) {
        this.prices = prices;
    }

    abstract calculate(): any;
}
