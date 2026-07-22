import jax
import jax.numpy as jnp
import pytest

from representation.primitives import Junction, WallSegment


class TestJunction:
    def test_position_stored_as_jax_array(self):
        j = Junction(id=0, position=(1.0, 2.0))
        assert isinstance(j.position, jnp.ndarray)
        assert j.position.shape == (2,)
        assert jnp.allclose(j.position, jnp.array([1.0, 2.0]))

    def test_x_y_properties(self):
        j = Junction(id=0, position=(3.0, 4.0))
        assert jnp.allclose(j.x, 3.0)
        assert jnp.allclose(j.y, 4.0)

    def test_pytree_roundtrip(self):
        j = Junction(id=7, position=(1.0, 2.0))
        leaves, treedef = jax.tree_util.tree_flatten(j)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.id == 7
        assert jnp.allclose(rebuilt.position, j.position)

    def test_id_is_static_aux_data(self):
        j = Junction(id=7, position=(1.0, 2.0))
        leaves, _ = jax.tree_util.tree_flatten(j)
        assert len(leaves) == 1
        assert jnp.allclose(leaves[0], j.position)

    def test_grad_flows_through_position(self):
        def loss(junction: Junction) -> jnp.ndarray:
            return jnp.sum(junction.position**2)

        j = Junction(id=0, position=(3.0, 4.0))
        grad = jax.grad(loss)(j)
        assert jnp.allclose(grad.position, jnp.array([6.0, 8.0]))

    def test_vmap_over_positions(self):
        positions = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def make_junction(pos):
            return Junction(id=0, position=pos)

        junctions = jax.vmap(make_junction)(positions)
        assert junctions.position.shape == (3, 2)


class TestWallSegment:
    def test_rejects_self_loop(self):
        with pytest.raises(ValueError):
            WallSegment(id=0, junction_start=1, junction_end=1)

    def test_as_array_resolves_from_junctions(self):
        junctions = {
            0: Junction(id=0, position=(0.0, 0.0)),
            1: Junction(id=1, position=(3.0, 4.0)),
        }
        seg = WallSegment(id=0, junction_start=0, junction_end=1)
        assert jnp.allclose(seg.as_array(junctions), jnp.array([0.0, 0.0, 3.0, 4.0]))

    def test_length(self):
        junctions = {
            0: Junction(id=0, position=(0.0, 0.0)),
            1: Junction(id=1, position=(3.0, 4.0)),
        }
        seg = WallSegment(id=0, junction_start=0, junction_end=1)
        assert jnp.allclose(seg.length(junctions), 5.0)

    def test_other_junction(self):
        seg = WallSegment(id=0, junction_start=0, junction_end=1)
        assert seg.other_junction(0) == 1
        assert seg.other_junction(1) == 0

    def test_other_junction_raises_for_non_incident(self):
        seg = WallSegment(id=0, junction_start=0, junction_end=1)
        with pytest.raises(ValueError):
            seg.other_junction(2)

    def test_is_frozen(self):
        seg = WallSegment(id=0, junction_start=0, junction_end=1)
        with pytest.raises(AttributeError):
            seg.id = 99

    def test_is_hashable(self):
        seg_a = WallSegment(id=0, junction_start=0, junction_end=1)
        seg_b = WallSegment(id=0, junction_start=0, junction_end=1)
        assert hash(seg_a) == hash(seg_b)
