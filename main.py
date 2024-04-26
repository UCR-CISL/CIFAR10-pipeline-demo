import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
from wrn import WideResNet
from dataloader import get_cifar10_loader, get_svhn_loader
from test import load_checkpoint
import itertools

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import datetime

url = "http://54.198.188.202:8086"
token = "fq91vv3Z-_Ez-5zzRU1_27EXdRpf6uWVij4NXQgmvrfEuH8FZ4kWTxeq5p6GRgZR9UCQ-xQTP_ucCN_rtMoFLQ=="
org = "cisl"
bucket = "cisl-dev"

# Influx Client
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

def streaming_data_loader(data_loader):
    """Yield batches of data from the DataLoader."""
    for data in data_loader:
        yield data

def calculate_energy_scores(logits, temperature=1.0):
    """Calculate and return the energy scores for given logits."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

def evaluate_model_streaming(model, stream_loader, device, steps=1):
    """Evaluate the model on streaming data for a specified number of steps and return energy scores."""
    model.eval()
    energy_scores = []
    with torch.no_grad():
        for i, (images, _) in enumerate(stream_loader):
            if i >= steps:
                break
            images = images.to(device)
            logits = model(images)
            energy_scores.extend(calculate_energy_scores(logits.cpu()).numpy())
    return energy_scores

def update_plot(frame_index, cifar10_stream, svhn_stream, model, device, line1, line2, ax):
    """Update plot and print energy scores for CIFAR-10 and SVHN streams."""
    cifar10_scores = evaluate_model_streaming(model, cifar10_stream, device)
    svhn_scores = evaluate_model_streaming(model, svhn_stream, device)

    # Update KDE plots
    scores_range = np.linspace(min(cifar10_scores + svhn_scores), max(cifar10_scores + svhn_scores), 1000)
    kde_cifar10 = gaussian_kde(cifar10_scores)
    kde_svhn = gaussian_kde(svhn_scores)
    line1.set_data(scores_range, kde_cifar10(scores_range))
    line2.set_data(scores_range, kde_svhn(scores_range))
    ax.relim()
    ax.autoscale_view()

    points = []
    for score, density_cifar, density_svhn in zip(scores_range, kde_cifar10(scores_range), kde_svhn(scores_range)):
        print(score, density_cifar, density_svhn)
        points.append(Point("energy_score_distribution")
                      .tag("model", "cifar10")
                      .field("energy_score", score)
                      .field("density", density_cifar)
                      .time(datetime.datetime.utcnow(), WritePrecision.NS))
        points.append(Point("energy_score_distribution")
                      .tag("model", "svhn")
                      .field("energy_score", score)
                      .field("density", density_svhn)
                      .time(datetime.datetime.utcnow(), WritePrecision.NS))

    write_api.write(bucket, org, points)

    # Print energy scores continuously in the terminal
    print(f"Update {frame_index}: CIFAR-10 Energy Scores: {cifar10_scores}, SVHN Energy Scores: {svhn_scores}")

    return line1, line2

def write_to_influxdb(client, data, measurement_name):
    sequence = []
    for score, density in data:
        point = (
            Point(measurement_name)
            .field("score", float(score))
            .field("density", float(density))
            .time(datetime.datetime.utcnow(), WritePrecision.NS)
        )
        sequence.append(point)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket, org, sequence)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideResNet(depth=40, widen_factor=2, dropRate=0.3, num_classes=10).to(device)
    load_checkpoint(model, "checkpoint_epoch_199.pt")

    influxdb_client = InfluxDBClient(url=url, token=token, org=org)

    cifar10_loader = get_cifar10_loader(batch_size=10, train=False)
    svhn_loader = get_svhn_loader(batch_size=10, split='test')

    cifar10_stream = itertools.cycle(streaming_data_loader(cifar10_loader))
    svhn_stream = itertools.cycle(streaming_data_loader(svhn_loader))
    
    frame_index = 0
    try:
        #for frame_index in range(100):  # Arbitrary number of updates
        while True:
            #cifar10_scores = evaluate_model_streaming(model, cifar10_stream, device, steps=1)
            #svhn_scores = evaluate_model_streaming(model, svhn_stream, device, steps=1)
            
            cifar10_scores = evaluate_model_streaming(model, cifar10_stream, device, steps=1)
            svhn_scores = evaluate_model_streaming(model, svhn_stream, device, steps=1)

            # Generate KDE for CIFAR-10 and SVHN
            scores_range = np.linspace(min(cifar10_scores + svhn_scores), max(cifar10_scores + svhn_scores), 1000)
            kde_cifar10 = gaussian_kde(cifar10_scores)
            kde_svhn = gaussian_kde(svhn_scores)
            
            # Prepare and write data to InfluxDB for CIFAR-10
            cifar10_data = list(zip(scores_range, kde_cifar10(scores_range)))
            write_to_influxdb(influxdb_client, cifar10_data, 'cifar10_kde')
            
            # Prepare and write data to InfluxDB for SVHN
            svhn_data = list(zip(scores_range, kde_svhn(scores_range)))
            write_to_influxdb(influxdb_client, svhn_data, 'svhn_kde')

            print(f"Frame {frame_index} data written to InfluxDB")
            frame_index += 1

    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == '__main__':
    main()
