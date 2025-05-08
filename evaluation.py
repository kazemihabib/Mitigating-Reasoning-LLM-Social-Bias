import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.function
def prep():
    pass


@app.function
def accuracy():
   # Accuracy = (Number of Correct Answers / Total Questions) × 100
    pass


@app.function
def bias_dis():
    #sDIS=2×( Number of non-UNKNOWN answers / Number of biased answers)−1
    pass


@app.function
def bias_amb():
    #sAMB=(1−accuracy)×sDIS
    pass


if __name__ == "__main__":
    app.run()
