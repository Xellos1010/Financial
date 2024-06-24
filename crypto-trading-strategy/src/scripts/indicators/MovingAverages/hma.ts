// src/scripts/indicators/MovingAverages/hma.ts
import { MovingAverageBase } from './movingAverageBase';
import { MovingAverageOutput } from '../types/movingAverages';
import { WMA } from './wma';

export class HMA extends MovingAverageBase {
    calculate(): MovingAverageOutput {
        const halfLength = Math.floor(this.period / 2);
        const sqrtLength = Math.floor(Math.sqrt(this.period));

        const wmaHalf = new WMA({ prices: this.prices, period: halfLength }).calculate().result;
        const wmaFull = new WMA({ prices: this.prices, period: this.period }).calculate().result;

        const diff = wmaHalf.map((val, idx) => 2 * val - wmaFull[idx]);

        const result = new WMA({ prices: diff, period: sqrtLength }).calculate().result;
        return { result };
    }
}
