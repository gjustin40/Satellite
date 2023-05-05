# metrics = {
#     'dice': 0.1,
#     'iou': 0.2
# }
# interval = 200
# msg = (
#     f'[{200}/{600}] | '
#     f'LR: {0.80:0.8e} | '
#     f'Loss: {0.7:0.4f} | '
#     f'Time: {0.0} | '
#     f'ETA: {0}'
# )

# print(msg)


metrics = {
    'dice': 0.1,
    'iou': 0.2
}

# Convert the metrics dictionary to a formatted string
# metrics_str = " | ".join([f"{key}: {value:.2f}" for key, value in train_avg.items()])

msg = (
    f'[{200}/{600}] | '
    f'LR: {0.80:0.8e} | '
    f'Loss: {0.7:0.4f} | '
    f'{" | ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])} | '
    f'Time: {0.0} | '
    f'ETA: {0} | '
)

print(msg)