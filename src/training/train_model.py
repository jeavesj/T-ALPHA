from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl

from src.models.full_model import MetaModel
from src.data.full_model_dataset import MetaModelDataset
from src.training.lightning_module import MetaModelLightning


def train_model(
    device="cuda",
    train_set=None,
    val_set=None,
    batch_size=32,
    num_epochs=120,
    batch_norm=True,
    patience=100,
    n_epochs=120,
    checkpoint_dir=None,
    save_dir=None,
    save_name=None,
    model_checkpoint_path=None,
    use_protein_graph=True,
    use_protein_surface=True,
    use_protein_sequence=True,
    use_ligand_properties=True,
    use_ligand_graph=True,
    use_ligand_sequence=True,
    use_complex_graph=True,
):
    """ "
    Trains the MetaModel using PyTorch Lightning with checkpointing and logging.

    Args:
        device (str, optional): Device for training ('cuda' or 'cpu'). Defaults to 'cuda'.
        train_set (str, optional): Path to the HDF5 training dataset.
        val_set (str, optional): Path to the HDF5 validation dataset.
        batch_size (int, optional): Batch size for training and validation. Defaults to 32.
        num_epochs (int, optional): Total number of training epochs. Defaults to 120.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        patience (int, optional): Early stopping patience. Defaults to 100.
        n_epochs (int, optional): Number of epochs for training. Defaults to 120.
        checkpoint_dir (str, optional): Directory to save model checkpoints.
        save_dir (str, optional): Directory to save the experiment logs.
        save_name (str, optional): Name for the experiment logs.
        model_checkpoint_path (str, optional): Path to a pre-trained model checkpoint. Defaults to None.
        optimizer_checkpoint_path (str, optional): Path to an optimizer checkpoint. Defaults to None.
    """

    # Initialize the model (do not move it to device yet)
    model = MetaModel(
        device=device,
        use_protein_graph=use_protein_graph,
        use_protein_surface=use_protein_surface,
        use_protein_sequence=use_protein_sequence,
        use_ligand_properties=use_ligand_properties,
        use_ligand_graph=use_ligand_graph,
        use_ligand_sequence=use_ligand_sequence,
        use_complex_graph=use_complex_graph,
        batch_norm=batch_norm,
    ).to(device)

    train_dataset = MetaModelDataset(train_set, device)
    val_dataset = MetaModelDataset(val_set, device)

    lightning_model = MetaModelLightning(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        batch_norm=batch_norm,
        patience=patience,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-{epoch:02d}",
        save_top_k=1,
        mode="max",
        monitor="val_correlation",
        verbose=True,
    )

    # Create a CSVLogger with custom options
    logger = CSVLogger(
        save_dir=save_dir,
        name=save_name,
    )

    trainer = pl.Trainer(
        devices=4,
        strategy="fsdp",
        accelerator="cuda",
        max_epochs=n_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=0.1,
        gradient_clip_algorithm="value",
    )

    if model_checkpoint_path is not None:
        print(f"Loading model from {model_checkpoint_path}")
        trainer.fit(lightning_model, ckpt_path=model_checkpoint_path)

    else:
        trainer.fit(lightning_model)
