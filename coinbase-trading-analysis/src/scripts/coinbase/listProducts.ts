
// src/scripts/coinbase/listProducts.ts
import axios from 'axios';

const url = 'https://api.coinbase.com/api/v3/brokerage/market/products';

export async function listProducts() {
    try {
        const response = await axios.get(url);
        const products = response.data.products;

        return products;
    } catch (error) {
        console.error('Error fetching products:', error);
        return [];
    }
}

export function filterPairings(products: any[], currency: string): string[] {
    return products
        .filter(product => {
            const baseCurrency = product.base_currency_id.toUpperCase();
            const quoteCurrency = product.quote_currency_id.toUpperCase();
            return (baseCurrency === currency.toUpperCase() || quoteCurrency === currency.toUpperCase());
        })
        .map(product => `${product.base_currency_id}-${product.quote_currency_id}`)
        .filter(isValidPairing);
}

function isValidPairing(pairing: string): boolean {
    const regex = /^[A-Z]{3,4}-[A-Z]{3,4}$/;
    return regex.test(pairing);
}