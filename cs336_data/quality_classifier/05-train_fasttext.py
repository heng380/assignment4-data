import fasttext

def main():
    model = fasttext.train_supervised(
        input = "/home/ubuntu/repos/assignment4-data/cs336_data/quality_classifier/training.txt",
        epoch = 30,
        lr=0.2
    )
    model.save_model("./quality.bin")
    print (model.test("./val.txt", k=1))

if __name__ == "__main__":
    main()





if __name__ == "__main__":
    main()