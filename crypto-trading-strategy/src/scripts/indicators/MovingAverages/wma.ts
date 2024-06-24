import { MovingAverageBase } from './movingAverageBase';
import { MovingAverageOutput } from '../types/movingAverages';

export class WMA extends MovingAverageBase {
    calculate(): MovingAverageOutput {
        let wma: number[] = [];
        const divisor = (this.period * (this.period + 1)) / 2;

        for (let i = 0; i <= this.prices.length - this.period; i++) {
            let sum = 0;
            for (let j = 0; j < this.period; j++) {
                sum += this.prices[i + j] * (this.period - j);
            }
            wma.push(sum / divisor);
        }

        return { result: wma };
    }
}
