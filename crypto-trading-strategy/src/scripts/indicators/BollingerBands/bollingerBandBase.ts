// src/scripts/indicators/BollingerBands/bollingerBandBase.ts

import { IndicatorBase } from '../indicatorBase';

export abstract class BollingerBandBase<T> extends IndicatorBase {
    protected period: number;
    protected multiplier: number;

    constructor(prices: number[], period: number, multiplier: number) {
        super(prices);
        this.period = period;
        this.multiplier = multiplier;
    }

    abstract calculateBands(): T;
}
