// src/scripts/indicators/MovingAverages/vwma.ts
import { MovingAverageBase } from './movingAverageBase';
import { MovingAverageOutput } from '../types/movingAverages';

export class VWMA extends MovingAverageBase {
    calculate(): MovingAverageOutput {
        let vwma: number[] = [];
        let cumulativePriceVolume = 0;
        let cumulativeVolume = 0;

        for (let i = 0; i < this.prices.length; i++) {
            cumulativePriceVolume += this.volumes![i] * this.prices[i];
            cumulativeVolume += this.volumes![i];
            vwma.push(cumulativePriceVolume / cumulativeVolume);
        }

        return { result: vwma };
    }
}
