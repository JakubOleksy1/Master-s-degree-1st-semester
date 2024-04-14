from result import Ok, Error

class BankAccount:
    def __init__(self):
        self.balance = 0

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if(self.balance > amount):
            self.balance -= amount
            return Ok("Here is your money", amount)

        return Error("ValueError('Insufficient funds')", amount)

    def __str__(self):
        return f"{self.balance} in bank account."
    
class MinimumBalanceAccount(BankAccount):
    def __init__(self, balance = 0, minimumBalance = 1000):
        super().__init__(balance)
        self.minimumBalance = minimumBalance

    def withdraw(self, amount):
        if(self.balance - amount > self.minimumBalance):
            return super().withdraw(amount)
        else:
            return Error("ValueError('Insufficient funds')", amount)

    def __str__(self):
        return f"{self.balance} in bank account with minimum balance of {self.minimumBalance}."
            