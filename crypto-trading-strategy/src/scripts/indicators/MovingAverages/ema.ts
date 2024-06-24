// src/scripts/indicators/MovingAverages/ema.ts
import { MovingAverageBase } from './movingAverageBase';
import { MovingAverageOutput } from '../types/movingAverages';

export class EMA extends MovingAverageBase {
    calculate(): MovingAverageOutput {
        let ema: number[] = [];
        const k = 2 / (this.period + 1);
        let sum = this.prices.slice(0, this.period).reduce((acc, val) => acc + val, 0);
        let firstEMA = sum / this.period;
        ema.push(firstEMA);

        for (let i = this.period; i < this.prices.length; i++) {
            let currentEMA = this.prices[i] * k + ema[ema.length - 1] * (1 - k);
            ema.push(currentEMA);
        }

        return { result: ema };
    }
}
