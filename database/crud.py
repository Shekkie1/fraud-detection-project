from database.models import Transaction


def create_transaction(
    db,
    amount,
    prediction,
    probability
):

    transaction = Transaction(
        amount=amount,
        prediction=prediction,
        probability=probability
    )

    db.add(transaction)

    db.commit()

    db.refresh(transaction)

    return transaction


def get_transactions(
    db
):

    return (
        db.query(Transaction)
        .order_by(
            Transaction.id.desc()
        )
        .all()
    )