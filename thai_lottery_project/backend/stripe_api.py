import stripe

stripe.api_key = "your-secret-key"

def create_checkout_session():
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price_data': {
                'currency': 'thb',
                'product_data': {'name': 'วิเคราะห์หวย'},
                'unit_amount': 10000,
            },
            'quantity': 1,
        }],
        mode='payment',
        success_url='http://localhost:8000/',
        cancel_url='http://localhost:8000/',
    )
    return {"url": session.url}
