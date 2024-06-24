import { MovingAverageInput, MovingAverageOutput } from '../types/movingAverages';

export abstract class MovingAverageBase {
    protected prices: number[];
    protected period: number;
    protected volumes: number[]; // Ensure volumes is always an array

    constructor(input: MovingAverageInput) {
        this.prices = input.prices;
        this.period = input.period;
        this.volumes = input.volumes || []; // Provide a default empty array if volumes are undefined
    }

    abstract calculate(): MovingAverageOutput;
}
