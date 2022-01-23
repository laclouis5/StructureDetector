from library import *
from torch.utils.data import DataLoader

from library.model.evaluator import Evaluations


class NetworkMock:

    def __init__(self) -> None:
        pass

    def __call__(self, encoded_data: dict) -> dict[str, torch.Tensor]:
        heatmaps = encoded_data["heatmaps"]  # (B, M, H/R, W/R)
        offsets = encoded_data["offsets"]  # (B, K, 2)
        embeddings = encoded_data["embeddings"]  # (B, K, 2)
        off_mask = encoded_data["off_mask"]  # (B, K)
        emb_mask = encoded_data["emb_mask"]  # (B, K)
        inds = encoded_data["inds"]  # (B, K)

        bs, _, h_r, w_r = heatmaps.shape

        out_offset = torch.zeros(bs, 2, h_r, w_r)
        out_embeddings = torch.zeros(bs, 2, h_r, w_r)

        for b in range(bs):
            for raw_index, off, is_included in zip(inds[b], offsets[b], off_mask[b]):
                if is_included:
                    x_r = raw_index % w_r
                    y_r = torch.div(raw_index, w_r, rounding_mode="floor")

                    out_offset[b, :, y_r, x_r] = off

            for raw_index, emb, is_included in zip(inds[b], embeddings[b], emb_mask[b]):
                if is_included:
                    x_r = raw_index % w_r
                    y_r = torch.div(raw_index, w_r, rounding_mode="floor")

                    out_embeddings[b, :, y_r, x_r] = emb

        return {
            "heatmaps": heatmaps,
            "offsets": out_offset,
            "embeddings": out_embeddings}


def main():
    args = Arguments().parse()
    assert args.valid_dir, "Path to a directory with validation samples must be specified."

    dataset = Dataset(args.valid_dir, TrainAugmentation(args))
    dataset = DataLoader(dataset, collate_fn=Dataset.collate_fn)

    net_mock = NetworkMock()
    
    # Loss
    # loss_fn = Loss(args)

    data = next(iter(dataset))
    out = net_mock(data)

    # _ = loss_fn(input=out, target=data)
    # print(loss_fn.stats)

    # Decoder
    decoder = Decoder(args)
    graph = decoder(out)[0]
    # im = data["image"][0]
    # im = un_normalize(im)
    # im = F.to_pil_image(im)
    # im = draw_graph(im, graph)
    # im.show()

    # Evaluator
    evaluator = Evaluator(args)
    gt = data["annotation"][0].to_graph()
    
    img_size = gt.image_size
    net_out_size = args.width, args.height
    image_path = gt.image_path

    gt.resize(net_out_size, img_size)
    pred = GraphAnnotation(image_path, graph, image_path)
    pred.resize(net_out_size, img_size)

    evaluator.evaluate_keypoints(pred, gt)
    evaluator.pretty_print()


if __name__ == "__main__":
    main()
