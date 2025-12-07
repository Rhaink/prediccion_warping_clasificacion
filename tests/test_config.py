"""
Tests para configuracion Hydra del proyecto.

Valida que los archivos de configuracion YAML sean correctos
y puedan cargarse con Hydra/OmegaConf.
"""

import pytest
from pathlib import Path

from omegaconf import OmegaConf


class TestConfigFilesExist:
    """Tests para verificar que archivos de config existen."""

    def test_config_yaml_exists(self, conf_path):
        """config.yaml principal debe existir."""
        assert (conf_path / 'config.yaml').exists()

    def test_model_config_exists(self, conf_path):
        """Configuracion de modelo debe existir."""
        assert (conf_path / 'model' / 'resnet18.yaml').exists()

    def test_training_config_exists(self, conf_path):
        """Configuracion de entrenamiento debe existir."""
        assert (conf_path / 'training' / 'default.yaml').exists()

    def test_data_config_exists(self, conf_path):
        """Configuracion de datos debe existir."""
        assert (conf_path / 'data' / 'default.yaml').exists()


class TestConfigLoading:
    """Tests para cargar configuraciones."""

    def test_load_main_config(self, conf_path):
        """Debe poder cargar config.yaml."""
        config_file = conf_path / 'config.yaml'
        cfg = OmegaConf.load(config_file)
        assert cfg is not None

    def test_load_model_config(self, conf_path):
        """Debe poder cargar model/resnet18.yaml."""
        config_file = conf_path / 'model' / 'resnet18.yaml'
        cfg = OmegaConf.load(config_file)
        assert cfg is not None

    def test_load_training_config(self, conf_path):
        """Debe poder cargar training/default.yaml."""
        config_file = conf_path / 'training' / 'default.yaml'
        cfg = OmegaConf.load(config_file)
        assert cfg is not None

    def test_load_data_config(self, conf_path):
        """Debe poder cargar data/default.yaml."""
        config_file = conf_path / 'data' / 'default.yaml'
        cfg = OmegaConf.load(config_file)
        assert cfg is not None


class TestMainConfigContent:
    """Tests para contenido de config.yaml principal."""

    @pytest.fixture
    def main_config(self, conf_path):
        """Carga config.yaml."""
        return OmegaConf.load(conf_path / 'config.yaml')

    def test_has_defaults(self, main_config):
        """Debe tener seccion defaults."""
        assert 'defaults' in main_config

    def test_has_project_section(self, main_config):
        """Debe tener seccion project."""
        assert 'project' in main_config

    def test_has_paths_section(self, main_config):
        """Debe tener seccion paths."""
        assert 'paths' in main_config

    def test_has_seed(self, main_config):
        """Debe tener seed para reproducibilidad."""
        assert 'seed' in main_config
        assert isinstance(main_config.seed, int)

    def test_has_device(self, main_config):
        """Debe tener configuracion de device."""
        assert 'device' in main_config

    def test_paths_has_required_keys(self, main_config):
        """Paths debe tener las claves requeridas."""
        required = ['data_root', 'csv_path', 'checkpoint_dir', 'output_dir']
        for key in required:
            assert key in main_config.paths, f"Falta paths.{key}"


class TestModelConfigContent:
    """Tests para contenido de model/resnet18.yaml."""

    @pytest.fixture
    def model_config(self, conf_path):
        """Carga model/resnet18.yaml."""
        return OmegaConf.load(conf_path / 'model' / 'resnet18.yaml')

    def test_has_name(self, model_config):
        """Debe tener nombre del modelo."""
        assert 'name' in model_config

    def test_has_architecture(self, model_config):
        """Debe tener seccion architecture."""
        assert 'architecture' in model_config

    def test_architecture_has_backbone(self, model_config):
        """Architecture debe especificar backbone."""
        assert 'backbone' in model_config.architecture
        assert model_config.architecture.backbone == 'resnet18'

    def test_has_head_config(self, model_config):
        """Debe tener configuracion de cabeza."""
        assert 'head' in model_config

    def test_has_output_config(self, model_config):
        """Debe tener configuracion de salida."""
        assert 'output' in model_config
        assert model_config.output.num_landmarks == 15
        assert model_config.output.num_coordinates == 30


class TestTrainingConfigContent:
    """Tests para contenido de training/default.yaml."""

    @pytest.fixture
    def training_config(self, conf_path):
        """Carga training/default.yaml."""
        return OmegaConf.load(conf_path / 'training' / 'default.yaml')

    def test_has_two_phase_flag(self, training_config):
        """Debe indicar si es entrenamiento en dos fases."""
        assert 'two_phase' in training_config

    def test_has_phase1_config(self, training_config):
        """Debe tener configuracion de fase 1."""
        assert 'phase1' in training_config

    def test_has_phase2_config(self, training_config):
        """Debe tener configuracion de fase 2."""
        assert 'phase2' in training_config

    def test_phase1_has_epochs(self, training_config):
        """Phase1 debe tener epochs."""
        assert 'epochs' in training_config.phase1
        assert training_config.phase1.epochs > 0

    def test_phase2_has_epochs(self, training_config):
        """Phase2 debe tener epochs."""
        assert 'epochs' in training_config.phase2
        assert training_config.phase2.epochs > 0

    def test_phase1_has_lr(self, training_config):
        """Phase1 debe tener learning rate."""
        assert 'learning_rate' in training_config.phase1
        assert training_config.phase1.learning_rate > 0

    def test_phase2_has_differential_lr(self, training_config):
        """Phase2 debe tener LR diferenciado para backbone y head."""
        assert 'backbone_lr' in training_config.phase2
        assert 'head_lr' in training_config.phase2
        # Backbone LR debe ser menor que head LR
        assert training_config.phase2.backbone_lr < training_config.phase2.head_lr

    def test_has_loss_config(self, training_config):
        """Debe tener configuracion de loss."""
        assert 'loss' in training_config
        assert 'type' in training_config.loss

    def test_has_checkpoint_config(self, training_config):
        """Debe tener configuracion de checkpoints."""
        assert 'checkpoint' in training_config


class TestDataConfigContent:
    """Tests para contenido de data/default.yaml."""

    @pytest.fixture
    def data_config(self, conf_path):
        """Carga data/default.yaml."""
        return OmegaConf.load(conf_path / 'data' / 'default.yaml')

    def test_has_image_size(self, data_config):
        """Debe tener tamano de imagen."""
        assert 'image_size' in data_config
        assert data_config.image_size == 224

    def test_has_splits(self, data_config):
        """Debe tener configuracion de splits."""
        assert 'splits' in data_config
        splits = data_config.splits
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

    def test_splits_sum_to_one(self, data_config):
        """Splits deben sumar 1.0."""
        splits = data_config.splits
        total = splits.train + splits.val + splits.test
        assert abs(total - 1.0) < 0.01

    def test_has_batch_size(self, data_config):
        """Debe tener batch size."""
        assert 'batch_size' in data_config
        assert data_config.batch_size > 0

    def test_has_preprocessing(self, data_config):
        """Debe tener configuracion de preprocesamiento."""
        assert 'preprocessing' in data_config

    def test_has_augmentation(self, data_config):
        """Debe tener configuracion de augmentation."""
        assert 'augmentation' in data_config


class TestConfigConsistency:
    """Tests de consistencia entre configs."""

    def test_model_landmarks_match_constants(self, conf_path):
        """Numero de landmarks en config debe coincidir con constantes."""
        from src_v2.constants import NUM_LANDMARKS, NUM_COORDINATES

        model_cfg = OmegaConf.load(conf_path / 'model' / 'resnet18.yaml')
        assert model_cfg.output.num_landmarks == NUM_LANDMARKS
        assert model_cfg.output.num_coordinates == NUM_COORDINATES

    def test_data_image_size_matches_constants(self, conf_path):
        """Tamano de imagen en config debe coincidir con constantes."""
        from src_v2.constants import DEFAULT_IMAGE_SIZE

        data_cfg = OmegaConf.load(conf_path / 'data' / 'default.yaml')
        assert data_cfg.image_size == DEFAULT_IMAGE_SIZE
