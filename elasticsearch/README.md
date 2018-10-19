### ElasticSearch on Kubernetes

This is a custom configuration for the [stable/elasticsearch helm chart](https://github.com/helm/charts/tree/master/stable/elasticsearch). 

#### Docker base image

A custom ElasticSearch base image with the [ElasticSearch Learning to Rank Plugin](https://github.com/o19s/elasticsearch-learning-to-rank/blob/master/docs/index.rst)
is defined in [elasticsearch-ltr.Dockerfile]. This is configured to be built by Docker Hub on each commit to the
repository. Additional plugins can be added to this Dockerfile as needed.

#### Installing Kubernetes and Helm

To establish parity between local development and production deployments, Kubernetes is used to create the
ElasticSearch cluster.
1. In the Docker-for-mac settings, enable Kubernetes support.
1. Ensure that you have `kubectl` installed on your path. Docker-for-mac adds this executable to 
`/usr/local/bin/kubectl`, or it can be installed with Homebrew.
1. Install Helm, the Kubernetes package manager. The easiest way to do this is with Homebrew.
    ```
    brew install kubernetes-helm
    ```
1. You may want to increase the resource allocation of the Docker install to have at least `4GB` of memory allocated.
1. Ensure that "Kubernetes is running" is displayed in the Docker-for-mac tray dropdown.
1. Ensure that `docker-for-desktop` is the current Kubernetes config by either selecting the option from the 
Kubernetes section of the Docker-for-mac tray dropdown, or by:
    ```bash
    kubectl config use-context docker-for-desktop
    ```

#### Installing ElasticSearch
After these packages are installed, you are ready to install and test the ElasticSearch cluster.

1. Navigate to the `elasticsearch` directory in the repo, the directory where this file is located.
1. Run the following command to add Tiller, the Helm server-side container, to the K8s cluster
    ```bash
    bin/helm_up
    ```
    this runs `helm init` and `helm update` to add Tiller and update the Helm repositories
1. Run the following command to start the ElasticSearch cluster.
    ```bash
    bin/start_es_local
    ```
    This will install the `stable/elasticsearch` chart using the value overrides specified in `values-local.yaml`.
    These overrides add configuration to ensure the health of the cluster when run locally, specify the use of the
    custom docker image, and bind a traffic to the client service's port `9200` to a random port. After this
    `start_es_local` sources `bin/set_port_env` to export this random port to the `NODE_PORT` environment variable.
1. Monitor the cluster with:
    ```bash
    kubectl get pods
    ```
    until there are 7 pods with 'ready' status: 2 client; 2 data; and 3 master pods.
    

#### Testing indexes
After the cluster is ready, run the test script to initialize the Learning to rank plugin and ingest data into
a test index

```bash
bin/test_index
```

This should not return any error messages.

#### Cleanup
To destroy the cluster and all volumes, run:

```bash
bin/clean
```

This removes the cluster, all PersistedVolumeClaims, and all PersistedVolumes.

Before reinstalling the cluster, ensure that all persisted volumes with a claim from the `elasticsearch-ltr` app
have been deleted. This can take some time but you may experience a nonfunctional cluster if the app is started too
soon.


####Other tools

##### ElasticSearch Head
[ElasticSearch Head](https://chrome.google.com/webstore/detail/elasticsearch-head/ffmkiejjmecolpfloofpjologoblkegm) is
a chrome extension that can be used to interact with ElasticSearch and to view the health of the cluster. It is 
a convenent way to deliver _ad hoc_ commands to the cluster.

##### Kitematic
[Kitematic](https://kitematic.com/) is a graphical frontend to Docker that is useful for exploring the logs of the 
running ElasticSearch cluster. It can be installed from its site or "Kitematic" menu item in the Docker-for-mac
tray menu. Pro tip: you can refresh the list of running containers using `<cmd> + R`.
