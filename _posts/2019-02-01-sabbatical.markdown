---
layout: post
title:  "Machine learning sabbatical"
---

At the start of 2018 I decided to pause work and take a sabbatical to study machine learning. I kept seeing ML algorithms generalize better and better across problems in different domains. This has made for an exciting progress in many fields, which also infected me. There is something magical in a learning algorithm — in seeing it perform better on a task, perceive an unseen pattern, generate new data, or operate an autonomous agent. That a sequence of matrix multiplications i.e. multiplying real numbers can produce complex skills or behaviors is about as unintuitive as Newtons third law and my chair pushing on me right now.	 Yet through effectively such transforms and optimization, you write the instructions for how a stream of training data (in itself, a probability distribution), should be molded, and you arrive at human-level ‘intelligence’ with respect to some skills eg object detection. Add the broad impact that AI has and this makes for something fascinating and should be learned.

This is a post to share my thinking and experience — why I took time off to dedicate to machine learning and what I learned along the way - from practically getting things working to concepts and topics I covered. There is no end to learning, so this post is also very much a work in progress and as such all comments and suggestions are welcomed.


### Why now

In short, not knowing *it* (ML, AI, data science) is becoming an existential problem. Looking over the past few years, we have very quickly learned to take some things for granted — of course an algorithm can recognize objects, of course an algorithm can generate realistic speech and images, of course an algorithm can play games better than people, of course an algorithm can drive a car as safer as we can. Somehow we have very quickly accepted changes in the status quo.

I’ve spent a lot of time in the world of healthcare — from both my parents being surgeons to a medical device business I helped build over the last years. In healthcare, the gravity of such ‘changes in the status quo’ is different than in other industries. Would you see a doctor, who on average … is average, or an algorithm which is say 10% better than average? For most products I may be ok with subaverage, in healthcare, when it comes to our own health or that of our close ones, it’s almost impossible to willingly go for subaverage. So when the same algorithm transfers across tasks in 5 different medical specialties and is on average better than physicians, that to me sounds existential for the status quo.

What makes it double existential is that I don’t think much of the industry has come to terms with what it can do with ML/AI. This spills across providers, health institutions, and pharma. A [recent post](https://moalquraishi.wordpress.com/2018/12/09/alphafold-casp13-what-just-happened/)  describing the AlphaFold success in protein folding summarized this well: “Take a problem proximal to tech companies’ bottom line, e.g. image recognition or speech, and imagine that no tech company was investing research money into the problem. (IBM alone has been working on speech for decades.) Then imagine that a pharmaceutical company suddenly enters ImageNet and blows the competition out of the water, leaving the academics scratching their heads at what just happened and the tech companies almost unaware it even happened.”

But how quickly would this happen? Many consider healthcare a late adopter - we have to make sure something works robustly before letting human health depend on it. Though a therapy/treatment is algorithmic and medications are effectively biochemical/biological algorithms, and both therapy and medications have been established and validated based on data and both are technically personalized in so far as my condition, history, disease, and response is unique enough that a physician has to provide it one-on-one for me specifically. So it’s not like people in the industry have never seen an algorithm.

The step change — the why now — in healthcare has been medical data going a/ digital, b/ massive, c/ personal, that is a combination of electronic health records, biobanks, and of course mobile phones. In short, data has gone from mostly static, where datasets are carefully collected with a specific hypothesis in mind that they then aim to answer, to live, where we have a stream of data, a 'live' dataset, and so a 'live' hypothesis space. This speeds up iteration cycles and is a step change in the pace of evolution of the industry.


### Why this is interesting

Going from static to live data and hypothesis space for me is the difference between a spark and an electrical current — as we learned to modulate electrical current we lit the lightbulb and after a few decades of iteration powered a computer. As we learned to control current, we abstracted away from the actual stream of electrons and its properties. We started from power generation and slowly abstracting away, advanced power generation to electrical current, electrical to data packets, data to information. We took 50 or so years to really get things up and going in each of these. So today, we live in a time where the stream of electrons is abstracted to a stream of information and we are just beginning to learn how to manipulate this information stream and the probability distributions that live on it.

So it is curious to think. Let's say the culmination of this 'routing a stream of electrons to do what we want it to do' thus far is silicon circuits. We are today at roughly the performance limits of how thin a silicon circuit can be with just a few layers of silicon atoms separating channels - Intel is mass producing 14nm chips and a few years back IBM announced 7nm technology. You pull your molecular ruler and you find out that a single strand of DNA is 2.5nm and a red blood cell 7,500nm ([more here](https://phys.org/news/2015-08-silicon-limits-power-electronics-revolution.html)). That is amazing precision. If we abstract the current powering chips away to the information stream that they support, we are learning to generate images, sound, speech, text, graphs, abstract data. We are learning to construct and manipulate complex, non-analytic probability distributions and information manifolds. It is exciting to think about where this ability to transform electrons to information can get us if we reach the precision we have guiding current on silicon chips.


### Now some practicals

1. Starting point
2. Setup - hardware, os, text editors
3. Programming languages and frameworks
4. Curriculum
5. Useful conferences


#### Starting point

When I decided on the sabbatical I had a good math foundation but almost extinct CS knowledge. On the side of math — I spent the previous 3-4 years covering the undergrad and early grad math curriculum at MIT carving about 4-5 hours a day around work. That was a great theoretical prep but all of it was pencil on paper and doing math with a computer is different. I had also stayed away from probability theory, optimization and numerical methods, being more interested in algebra and topology. Both had learning curves. On the side of CS — I had some knowledge of Matlab and had mostly forgotten all of visual basic and java I had done an eon in high school. This was another learning curve.


#### Setup

Data science and traditional machine learning can be done on your laptop for the most part. For deep learning, though, you really need a GPU — the graphics card on your computer which is optimized for parallel computations. The difference can be a 10x speedup in training a model on a single GPU vs on CPU.  

I use a Mac and Google’s cloud compute engine. Mac OS uses a Unix kernel and Ubuntu is the most popular Linux OS for cloud virtual machines (also build on a Unix kernel) so this makes going from a Mac to a cloud instance relatively comparable. On the cloud, you can scale storage and compute (number of CPU cores / GPUs) in a second and Google has very good preemptible instance prices, so I decided against building my own workstation or getting an external GPU (since then, Google launched Colab which has free GPUs; though at first it was tricky to get it working reliably). Setting up a cloud instance with all necessary drivers and frameworks can be a pain — it took a day to get an Ubuntu instance to work with Cuda, the Nvidia GPU drivers. Although this is now down to a few minutes, since Google released ‘data science’ virtual instances, which come with all the drivers and frameworks installed.

[Docker](https://www.docker.com) is a useful tool if you need portability — it allows you to basically image your whole environment (os, applications, packages, dependencies) and replicate it other machines. Conversely, you can run someone else’s Docker image directly, so you don’t have to worry about replicating their setup to run their code but focus on their results.

Text editors are something else you’ll need to set up — this is where you write code. I use [vim](https://macvim-dev.github.io/macvim) for larger models and [jupyter](https://jupyter.org) notebooks for more interactive code. Vim comes with every Unix machine, which makes your life easy when you switch between cloud instances and your local machine (though has a bit of a learning curve). I use the [Janus](https://github.com/carlhuda/janus) distribution of vim, which packages plugins for editing, navigation, visualization and so on. Jupyter, on the other hand, is great for interactive code — you can see the output of a snippet of code immediately below your input. This makes tweaking code and making it work very quick. For a larger codebase/model with multiple modules, though, I find vim cleaner, easier and faster. I also tried Sublime and some development environments, but nothing beats vim.


#### Programming languages and frameworks

[Python](https://www.python.org) is the most popular language for machine learning. The [anaconda distribution](https://www.anaconda.com/distribution/) is the standard base on which you add libraries for numerical operations, scientific computing, plotting and visualization, computer vision, automatic differentiation, etc. Among the must-learn libraries are Numpy (numerical operations), Scipy (scientific computing), Matplotlib and Seaborn (visualization), OpenCV (computer vision), Scikit learn (ML algorithms). Many of the python libraries run on C++ cores or have extensions in C++ so learning to read and write C can be very useful, but not necessary to start with or to run research code. There are many older machine learning libraries written in Matlab (Matlab is mostly used in academia), so learning a bit of Matlab can also be useful.

[R](https://www.r-project.org) is another very popular language used in statistics and data science. R makes many traditional statistical learning algorithms very quick to run and tweak (a few lines of code), but lacks the support and codebases when it comes to deep learning.

Bash is a scripting language used to access the OS and filesystem and execute code. It saves a lot of time when running larger models which require multi-step execution, e.g. multiple functional modules or data pre- or post-processing pipelines, integrating code with background OS processes, or running things on-/off-cloud efficiently. Learning a bit of bash scripting can save a ton of time ([some helpful videos here](https://www.youtube.com/playlist?list=PLAn5BRyzQEf9VoK8gRKp8Z0LGME6fISaE)

**Deep learning frameworks**. The world here has pretty much divided in two — [Tensorflow](https://www.tensorflow.org) (Google) and [Pytorch](https://pytorch.org) (Facebook). Tensorflow generally is the go-to for production code, and Pytorch is widely used for research but has recently incorporated Caffe2 to facilitate production. My preference is Pytorch — models are built in a more intuitive, ‘linear’ fashion, there is one way to do things (vs multiple sublibraries implementing the same functionality in Tensorflow), it is very close to Numpy and easier to debug (you are debugging the model logic vs. why / why not some component library works in my experience with Tensorflow). In short, I found I was iterating much faster on Pytorch so I mostly use it. There are many models with only Tensorflow reference implementations, however, so you need to know both. Theano is an older framework still used in research and I have found myself looking through Theano code for references quite a few times.

**Probabilistic programming**. These are more specialized frameworks for probabilistic inference, statistical modeling, and sampling methods, that is when you can model a data generative process e.g. via a graphical model. There are many frameworks based on different core language. For python, recent projects are Edward (Columbia; built on Tensorflow) and Pyro (built on Pytorch) and older/more established ones are Microsoft’s InferNET, Stan and PyMC3.

It is useful to execute all of the above in a virtual environment (depending on how portable you need to be, on a Docker image as well). For python, anaconda and virtualenv let you create virtual environments where you can freeze package versions and dependencies, and avoid conflict with the rest of your system or non-functional code when you update your system-wide installation to newer libraries.

Lastly, to track code changes, share code between local and cloud, or publish code you need to learn git and integrate with Github, Google Repos or other cloud code repo service.


#### Curriculum

I divided the curriculum in four: traditional statistical learning, natural language, vision, and reinforcement learning. I found courses, textbooks, curricula and video online, and online classes that would give me an overview of each area and a deeper dive. There were some math/CS preliminaries Most of these share problem sets with references for the metrics of a correct implementation (e.g. loss value, accuracy, precision, recall, etc), so you can implement on your own and check your work. Below are the resources I went through and found most helpful - my criteria were depth, rigor, complexity, and availability of additional resources.

**Math preliminaries**:
- Multivariable calculus, linear algebra, analysis and functional analysis I had covered. Things I added are:
- [MIT 6.041 Probabilistic systems analysis and applied probability](http://ocw.mit.edu/6-041SCF13)
- [MIT 18.650 Statistics for Applications](http://ocw.mit.edu/18-650F16)
- [MIT 6.006 Introduction to Algorithms](http://ocw.mit.edu/6-006F11)
- [CMU 10-725 Convex optimization](http://www.stat.cmu.edu/~ryantibs/convexopt/)
- [Stanford CS205 Mathematical Methods for Robotics, Vision, and Graphics](http://graphics.stanford.edu/courses/cs205a/)

**Machine learning**:
- [Murphy - Machine learning a probabilistic perspective](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/)
- [Bishop - Pattern recognition and machine learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738/)
- Hastie, Tibshirani, and Friedman - Elements of statistical learning — [videos](https://lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/about) and [book](https://web.stanford.edu/~hastie/ElemStatLearn/)
- Koller and Friedman - Probabilistic graphical models — [videos](http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=ProbabilisticGraphicalModels) and [book](https://mitpress.mit.edu/books/probabilistic-graphical-models)
- UBC 340 - Machine learning and data mining — [course](https://www.cs.ubc.ca/~nando/340-2012/index.php) and [videos](https://www.youtube.com/playlist?list=PLE6Wd9FR--Ecf_5nCbnSQMHqORpiChfJf)
- UBC 540 - Machine learning and data mining — [course](https://www.cs.ubc.ca/~nando/540-2013/index.html) and [videos](https://www.youtube.com/playlist?list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6)
- David MacKay’s Information theory, pattern recognition, and neural networks - [book](http://www.inference.org.uk/mackay/itila/book.html) and [lectures](https://www.youtube.com/playlist?list=PLruBu5BI5n4aFpG32iMbdWoRVAA-Vcso6)
- [Berkeley CS188 - Artificial Intelligence](http://ai.berkeley.edu/)
- [Sheffield Gaussian Processes summer school](http://gpss.cc/gpss18/program)
- [UMich Applied data science with python specialization](https://www.coursera.org/specializations/data-science-python)
- Further great resources are Harvard [CS181](https://harvard-ml-courses.github.io/cs181-web/) and [CS281](https://harvard-ml-courses.github.io/cs281-web/), and Stanford [CS229](http://cs229.stanford.edu)

**Deep learning**:
- [Andrew Ng’s Coursera specialization](https://www.coursera.org/specializations/deep-learning)
- CMU 10-707 - Topics in deep learning — [course](https://deeplearning-cmu-10707.github.io) and [videos](https://www.youtube.com/playlist?list=PLpIxOj-HnDsOSL__Buy7_UEVQkyfhHapa)
- [Stanford CS230 - Deep learning](https://cs230-stanford.github.io)

**NLP**:
- [Oxford Deep NLP](https://github.com/oxford-cs-deepnlp-2017)
- [Stanford CS224n](http://web.stanford.edu/class/cs224n/)
- [CMU 11-747 Neural nets for NLP](http://phontron.com/class/nn4nlp2018/index.html)

**Vision**:
- [Stanford CS231n](http://cs231n.stanford.edu)

**RL**:
- [Sutton & Barto - Reinforcement learning an introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Berkeley CS294 - Deep reinforcement learning](http://rail.eecs.berkeley.edu/deeprlcourse/)
- UCL course on RL — [course](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html) and [videos](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

Along the way I implemented course/text resources or open-source references on the topic - this is key to understand both theory and practice. Without theory, you don’t know what the thing you implement does or how it does it; without implementing, connecting theory on paper to code in your head is useless since this entire field is anchored on optimization, which has convexity, convergence, and numerical properties that are non-intuitive.

Each topic area has major / key papers that have marked progress in the field over the last 5-20 years. Reading the key papers in parallel with the latest research for each field helps you understand the relevance of the methods/concepts/algorithms and get a sense of progress and velocity.

Workshop and tutorials at conferences and institutes are another great resource to augment the above with most up to date models and methods. E.g. Simons institute for ML, CVPR for vision, ICML for ML, UAI for generative models and bayesian inference, and NeurIPS overall. I’ve added a list of useful conferences below.

Framework often comes with tutorials and doing these is quite helpful to both understand concepts and understand the framework itself. I found that for the more popular frameworks, the quality of the code in many examples isn’t very good — code works but but may still have mistakes and be a bit hacky. For speciality frameworks, e.g. Gaussian process or variational inference, I found the code examples generally higher quality. For clean and exemplar code, official open-source reference implementations of major models in the space are the best learning resource to augment, rebuild, or simply use — e.g. major vision models for detection, localization, segmentation; language and translation models, generative models.

I went deeper implementing a multiple papers are: (I find fascinating learning a model to dream up / generate unseen data or learning a probability distribution over eg an imaging dataset which caries semantic meaning)
- generative models — these are latent variable models, which assume that the data is generated by some random process involving an unobserved ie latent variable or variables;  e.g. variational autoencoders (VAEs), generative adversarial networks (GANS), adversarial autoencoders, autoregressive models.
- normalizing flows — these are models, which learn to transform a known base distribution (eg standard Gaussian) into a target density (eg face images) by invertible transformations.
- variational inference — this is a set of inference methods for approximating a posterior distribution, at the heart of generative models and normalizing flows
- object detection and localization — these are computer vision models, which detect and localize an object within an image regressing a bounding box, key points, or mask over the object.


#### Useful conferences

(with great paper presentations and tutorials on youtube):
- Uncertainty in artificial intelligence ([UAI](http://www.auai.org))
- International conference on machine learning ([ICML](https://icml.cc))
- Computer vision and pattern recognition ([CVPR](http://cvpr2018.thecvf.com))
- [SIGGRAPH](https://s2018.siggraph.org/)
- Neural information processing systems ([NeurIPS](https://neurips.cc))
- [Simons institute](https://simons.berkeley.edu) at UC Berkeley (has many public workshops).


#### What’s more to do

A year ago I first sought breadth first, depth second. Moving forward, this has to flip. Practically that means applying what I’ve learned to areas and problems I care about and make a difference in the world. Theoretically that means going deeper in the areas I find fascinating — computer vision, reinforcement learning, probabilistic programing, variational inference, information geometry.


### In closing

We live in a wonderful world today in having open access to information. When I applied to colleges (early 2000s in Bulgaria) I had to travel half an hour to an Open Society library where they had SAT books I could borrow; today, I pull up my computer and have access to courses, materials, and videos from the best universities in the world on the click of a button. This is something to cherish. I am grateful to MIT, Stanford, Berkeley, Harvard, CMU, Oxford, UCL, Sheffield, UBC, and the many others, the amazing professors, phd’s and post-docs at these institution for sharing and upkeeping all of their course and research materials and making it freely available to others.
