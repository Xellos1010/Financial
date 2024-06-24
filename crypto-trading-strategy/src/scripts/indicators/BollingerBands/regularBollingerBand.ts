// src/scripts/indicators/BollingerBands/regularBollingerBand.ts

import { BollingerBandBase } from './bollingerBandBase';
import { RegularBollingerBands } from '../types/bollingerBands';

export class RegularBollingerBand extends BollingerBandBase<RegularBollingerBands> {
    calculateBands(): RegularBollingerBands {
        const basis = this.calculateSMA(this.prices, this.period);
        const stdDev = this.calculateStdDev(basis);
        const upperBand = basis.map((b, i) => b + this.multiplier * stdDev[i]);
        const lowerBand = basis.map((b, i) => b - this.multiplier * stdDev[i]);

        return { upperBand, lowerBand, basis };
    }

    private calculateSMA(prices: number[], period: number): number[] {
        let sma: number[] = [];
        for (let i = 0; i < prices.length - period + 1; i++) {
            let sum = prices.slice(i, i + period).reduce((acc, val) => acc + val, 0);
            sma.push(sum / period);
        }
        return sma;
    }

    private calculateStdDev(basis: number[]): number[] {
        let stdDev: number[] = [];
        for (let i = 0; i < this.prices.length; i++) {
            let sum = 0;
            for (let j = i; j < i + this.period && j < this.prices.length; j++) {
                sum += Math.pow(this.prices[j] - basis[i], 2);
            }
            stdDev.push(Math.sqrt(sum / this.period));
        }
        return stdDev;
    }

    calculate(): RegularBollingerBands {
        return this.calculateBands();
    }
}
