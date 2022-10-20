## What is Multiple Instance Learning (MIL)?

Multiple Instance Learning is a type of weakly supervised learning algorithm
where training data is arranged in bags,
where each bag contains a set of instances and one single label.
MIL is most commonly used on very large whole slide images,
but it can be applied to any kind of large images that can't be processed by itself.

## MIL Wrapper

MIL wrapper is a very simple extension with an additional aggregation stage since inputs have an extra dimension.
First and second aggregation/pooling methods are taken from the MIL papers.
MIL wrapper is illustrated below.

![image info](static/diagram.png)

## References

* Gadermayr, Michael, and Maximilian Tschuchnig. "Multiple Instance Learning for Digital Pathology: A Review on the State-of-the-Art, Limitations & Future Potential." arXiv preprint arXiv:2206.04425 (2022).
* Ilse, Maximilian, Jakub Tomczak, and Max Welling. "Attention-based deep multiple instance learning." International conference on machine learning. PMLR, 2018.
* [Mayo Clinic - STRIP AI, 1st Place Solution](https://www.kaggle.com/competitions/mayo-clinic-strip-ai/discussion/357892)
* [Mayo Clinic - STRIP AI, 25th Place Solution](https://www.kaggle.com/competitions/mayo-clinic-strip-ai/discussion/357898)