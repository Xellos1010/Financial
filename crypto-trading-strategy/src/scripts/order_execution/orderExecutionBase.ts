// src/scripts/order_execution/orderExecutionBase.ts
export abstract class OrderExecutionBase {
    abstract executeBuyOrder(productId: string, price: number): void;
    abstract executeSellOrder(productId: string, price: number): void;
}
