from pydantic import BaseModel

class LoanApplication(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    delinq_2yrs: int
    inq_last_6mths: int
    open_acc: int
    pub_rec: int
    revol_bal: float
    revol_util: float
    total_acc: int
    mort_acc: int
    pub_rec_bankruptcies: int