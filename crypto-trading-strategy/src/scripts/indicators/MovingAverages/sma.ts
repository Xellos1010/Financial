// src/scripts/indicators/MovingAverages/sma.ts
import { MovingAverageBase } from './movingAverageBase';
import { MovingAverageOutput } from '../types/movingAverages';

export class SMA extends MovingAverageBase {
    calculate(): MovingAverageOutput {
        let sma: number[] = [];
        for (let i = 0; i < this.prices.length - this.period + 1; i++) {
            let sum = this.prices.slice(i, i + this.period).reduce((acc, val) => acc + val, 0);
            sma.push(sum / this.period);
        }
        return { result: sma };
    }
}
