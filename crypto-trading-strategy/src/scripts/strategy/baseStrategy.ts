// src/scripts/strategy/baseStrategy.ts
import { IndicatorBase } from '../indicators/indicatorBase';
import fetchAndSaveCandles from '../coinbase/getProductCandles';
import { OrderExecutionBase } from '../order_execution/orderExecutionBase';

interface Indicator {
    name: string;
    instance: IndicatorBase;
}

export abstract class BaseStrategy<T extends OrderExecutionBase> {
    protected indicators: Indicator[];
    protected productId: string;
    protected granularity: string;
    protected orderExecution: T;

    constructor(productId: string, granularity: string, orderExecution: T) {
        this.indicators = [];
        this.productId = productId;
        this.granularity = granularity;
        this.orderExecution = orderExecution;

        // Bind order execution methods
        this.executeBuyOrder = this.orderExecution.executeBuyOrder.bind(this.orderExecution);
        this.executeSellOrder = this.orderExecution.executeSellOrder.bind(this.orderExecution);
    }

    // Define the methods explicitly
    executeBuyOrder(productId: string, price: number): void {
        throw new Error('Method not implemented.');
    }

    executeSellOrder(productId: string, price: number): void {
        throw new Error('Method not implemented.');
    }

    addIndicator(name: string, indicator: IndicatorBase) {
        this.indicators.push({ name, instance: indicator });
    }

    async fetchCandles(startTime: number, endTime: number) {
        return await fetchAndSaveCandles(this.productId, this.granularity, startTime, endTime);
    }

    abstract generateSignals(candles: any[]): void;
}
