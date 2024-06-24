// src/scripts/indicators/MovingAverages/rma.ts
import { MovingAverageBase } from './movingAverageBase';
import { MovingAverageOutput } from '../types/movingAverages';

export class RMA extends MovingAverageBase {
    calculate(): MovingAverageOutput {
        let rma: number[] = [];
        let alpha = 1 / this.period;

        rma[0] = this.prices[0];
        for (let i = 1; i < this.prices.length; i++) {
            rma[i] = alpha * this.prices[i] + (1 - alpha) * rma[i - 1];
        }

        return { result: rma };
    }
}
