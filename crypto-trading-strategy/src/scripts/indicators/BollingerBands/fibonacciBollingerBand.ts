// src/scripts/indicators/BollingerBands/fibonacciBollingerBand.ts
import { BollingerBandBase } from './bollingerBandBase';
import { FibonacciBollingerBands } from '../types/bollingerBands';
import { getMovingAverage } from '../MovingAverages/movingAverageFactory';
import { MovingAverageType } from '../types/movingAverages';

export class FibonacciBollingerBand extends BollingerBandBase<FibonacciBollingerBands> {
    private source: number[];
    private maType: MovingAverageType;

    constructor(prices: number[], period: number, multiplier: number, source: number[], maType: MovingAverageType) {
        super(prices, period, multiplier);
        this.source = source;
        this.maType = maType;
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

    calculateBands(): FibonacciBollingerBands {
        const basis = getMovingAverage(this.maType, { prices: this.prices, period: this.period, volumes: this.source }).result;
        const stdDev = this.calculateStdDev(basis);
        const dev = stdDev.map((value) => value * this.multiplier);

        const upperBands = {
            upper_1: basis.map((b, i) => b + 0.236 * dev[i]),
            upper_2: basis.map((b, i) => b + 0.382 * dev[i]),
            upper_3: basis.map((b, i) => b + 0.5 * dev[i]),
            upper_4: basis.map((b, i) => b + 0.618 * dev[i]),
            upper_5: basis.map((b, i) => b + 0.764 * dev[i]),
            upper_6: basis.map((b, i) => b + dev[i]),
        };

        const lowerBands = {
            lower_1: basis.map((b, i) => b - 0.236 * dev[i]),
            lower_2: basis.map((b, i) => b - 0.382 * dev[i]),
            lower_3: basis.map((b, i) => b - 0.5 * dev[i]),
            lower_4: basis.map((b, i) => b - 0.618 * dev[i]),
            lower_5: basis.map((b, i) => b - 0.764 * dev[i]),
            lower_6: basis.map((b, i) => b - dev[i]),
        };

        return { basis, upperBands, lowerBands };
    }

    calculate(): FibonacciBollingerBands {
        return this.calculateBands();
    }
}
