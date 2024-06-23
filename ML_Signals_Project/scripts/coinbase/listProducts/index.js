const axios = require('axios');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

const url = 'https://api.coinbase.com/api/v3/brokerage/market/products';

axios.get(url)
    .then(response => {
        const products = response.data.products;
        const csvWriter = createCsvWriter({
            path: '../../../data/coinbase/products/products.csv',
            header: [
                { id: 'product_id', title: 'Product ID' },
                { id: 'price', title: 'Price' },
                { id: 'price_percentage_change_24h', title: 'Price Percentage Change 24h' },
                { id: 'volume_24h', title: 'Volume 24h' },
                { id: 'volume_percentage_change_24h', title: 'Volume Percentage Change 24h' },
                { id: 'base_increment', title: 'Base Increment' },
                { id: 'quote_increment', title: 'Quote Increment' },
                { id: 'quote_min_size', title: 'Quote Min Size' },
                { id: 'quote_max_size', title: 'Quote Max Size' },
                { id: 'base_min_size', title: 'Base Min Size' },
                { id: 'base_max_size', title: 'Base Max Size' },
                { id: 'base_name', title: 'Base Name' },
                { id: 'quote_name', title: 'Quote Name' },
                { id: 'watched', title: 'Watched' },
                { id: 'is_disabled', title: 'Is Disabled' },
                { id: 'new', title: 'New' },
                { id: 'status', title: 'Status' },
                { id: 'cancel_only', title: 'Cancel Only' },
                { id: 'limit_only', title: 'Limit Only' },
                { id: 'post_only', title: 'Post Only' },
                { id: 'trading_disabled', title: 'Trading Disabled' },
                { id: 'auction_mode', title: 'Auction Mode' },
                { id: 'product_type', title: 'Product Type' },
                { id: 'quote_currency_id', title: 'Quote Currency ID' },
                { id: 'base_currency_id', title: 'Base Currency ID' },
                { id: 'base_display_symbol', title: 'Base Display Symbol' },
                { id: 'quote_display_symbol', title: 'Quote Display Symbol' }
            ]
        });

        csvWriter.writeRecords(products)
            .then(() => {
                console.log('Products have been written to CSV file successfully.');
            });
    })
    .catch(error => {
        console.error('Error fetching products:', error);
    });
