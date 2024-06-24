// src/scripts/indicators/MovingAverages/movingAverageFactory.ts
import { SMA } from './sma';
import { EMA } from './ema';
import { DEMA } from './dema';
import { TEMA } from './tema';
import { VWMA } from './vwma';
import { ZLEMA } from './zlema';
import { WMA } from './wma';
import { HMA } from './hma';
import { RMA } from './rma';
import { MovingAverageType, MovingAverageInput, MovingAverageOutput } from '../types/movingAverages';

export const getMovingAverage = (type: MovingAverageType, input: MovingAverageInput): MovingAverageOutput => {
    switch (type) {
        case 'SMA':
            return new SMA(input).calculate();
        case 'EMA':
            return new EMA(input).calculate();
        case 'DEMA':
            return new DEMA(input).calculate();
        case 'TEMA':
            return new TEMA(input).calculate();
        case 'VWMA':
            return new VWMA(input).calculate();
        case 'ZLEMA':
            return new ZLEMA(input).calculate();
        case 'WMA':
            return new WMA(input).calculate();
        case 'HMA':
            return new HMA(input).calculate();
        case 'RMA':
            return new RMA(input).calculate();
        default:
            throw new Error('Unsupported moving average type');
    }
};
