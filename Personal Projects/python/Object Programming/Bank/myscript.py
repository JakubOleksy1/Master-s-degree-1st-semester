from bankaccount import BankAccount, MinimumBalanceAccount

""" bankAccount = BankAccount()
bankAccount.deposit(100)
result = bankAccount.withdraw(50)
result = bankAccount.withdraw(10) # This will raise an error

print(result.message)
if(result.isSuccess):
    print(result.value)
    print("All good")
else:
    print(result.value) """

accountMin = MinimumBalanceAccount(1500, 1000)
result = accountMin.withdraw(500)
