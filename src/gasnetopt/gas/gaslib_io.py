'''
Parsers for GasLib network files (.net, gaslib.zib.de) and TRR154 transient
data files (.bcd boundary conditions, .state initial states,
https://www.trr154.fau.de/transient-data/).

All quantities are converted to SI units on parsing:
pressure [Pa], length [m], mass flow [kg/s], temperature [K].
GasLib volumetric flow bounds (1000 m^3/h at norm conditions) are converted
to mass flow with the norm density given in the .net file.
'''

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# unit -> factor to SI
_UNIT_FACTORS = {
    'bar': 1e5,                  # -> Pa
    'Pa': 1.,
    'm': 1.,
    'km': 1e3,
    'mm': 1e-3,
    'kg_per_s': 1.,
    'm_per_s': 1.,
    's': 1.,
    'kg_per_m_cube': 1.,
    'K': 1.,
}


def _value(elem: ET.Element | None,
           default: float | None = None) -> float | None:
    'Read a value/unit attribute pair into SI units.'
    if elem is None:
        return default
    val = float(elem.get('value'))
    unit = elem.get('unit')
    if unit is None:
        return val
    if unit == 'Celsius':
        return val + 273.15
    if unit == '1000m_cube_per_hour':
        # volumetric flow at norm conditions; conversion to kg/s requires
        # the norm density and is done by the caller
        return val
    return val * _UNIT_FACTORS[unit]


def _local(tag: str) -> str:
    'Strip XML namespace from a tag.'
    return tag.rsplit('}', 1)[-1]


@dataclass
class Node:
    id: str
    type: str                    # 'source' | 'sink' | 'innode'
    height: float = 0.
    pressure_min: float = 0.     # Pa
    pressure_max: float = np.inf
    flow_min: float = 0.         # kg/s (sources/sinks)
    flow_max: float = np.inf
    x: float = 0.
    y: float = 0.


@dataclass
class Pipe:
    id: str
    from_node: str
    to_node: str
    length: float                # m
    diameter: float              # m
    roughness: float             # m
    flow_min: float = -np.inf    # kg/s
    flow_max: float = np.inf

    @property
    def area(self) -> float:
        return np.pi * self.diameter ** 2 / 4.

    def friction_factor(self) -> float:
        'Nikuradse friction factor for fully turbulent rough flow.'
        return (2. * np.log10(self.diameter / self.roughness) + 1.138) ** -2


@dataclass
class Valve:
    id: str
    from_node: str
    to_node: str
    flow_min: float = -np.inf            # kg/s
    flow_max: float = np.inf
    pressure_diff_max: float = np.inf    # Pa


@dataclass
class CompressorStation:
    id: str
    from_node: str
    to_node: str
    flow_min: float = 0.                 # kg/s
    flow_max: float = np.inf
    pressure_in_min: float = 0.          # Pa
    pressure_out_max: float = np.inf


@dataclass
class GasNetwork:
    name: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    pipes: List[Pipe] = field(default_factory=list)
    valves: List[Valve] = field(default_factory=list)
    compressors: List[CompressorStation] = field(default_factory=list)
    norm_density: float = 0.785          # kg/m^3

    @property
    def sources(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.type == 'source']

    @property
    def sinks(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.type == 'sink']

    @property
    def innodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.type == 'innode']

    @property
    def switches(self) -> list:
        'Switchable elements: valves and compressor stations (in this order).'
        return list(self.valves) + list(self.compressors)


def _flow_to_kg_per_s(elem, norm_density: float) -> float | None:
    'Convert a flow element (possibly in 1000m_cube_per_hour) to kg/s.'
    if elem is None:
        return None
    val = float(elem.get('value'))
    unit = elem.get('unit')
    if unit == '1000m_cube_per_hour':
        return val * 1000. / 3600. * norm_density
    if unit == 'kg_per_s':
        return val
    raise ValueError('Unsupported flow unit "{}"'.format(unit))


def parse_net(path) -> GasNetwork:
    'Parse a GasLib .net file into a GasNetwork (SI units).'
    root = ET.parse(path).getroot()
    title = None
    nodes_elem = conns_elem = None
    for child in root:
        tag = _local(child.tag)
        if tag == 'information':
            for e in child:
                if _local(e.tag) == 'title':
                    title = e.text
        elif tag == 'nodes':
            nodes_elem = child
        elif tag == 'connections':
            conns_elem = child

    net = GasNetwork(name=title or str(path))

    # first pass: norm density from any source node
    for e in nodes_elem:
        if _local(e.tag) == 'source':
            for c in e:
                if _local(c.tag) == 'normDensity':
                    net.norm_density = _value(c)
            break

    def child_map(elem):
        return {_local(c.tag): c for c in elem}

    for e in nodes_elem:
        tag = _local(e.tag)
        if tag not in ('source', 'sink', 'innode'):
            continue
        c = child_map(e)
        node = Node(
            id=e.get('id'), type=tag,
            height=_value(c.get('height'), 0.),
            pressure_min=_value(c.get('pressureMin'), 0.),
            pressure_max=_value(c.get('pressureMax'), np.inf),
            x=float(e.get('x', 0.)), y=float(e.get('y', 0.)),
        )
        if 'flowMin' in c:
            node.flow_min = _flow_to_kg_per_s(c['flowMin'], net.norm_density)
        if 'flowMax' in c:
            node.flow_max = _flow_to_kg_per_s(c['flowMax'], net.norm_density)
        net.nodes[node.id] = node

    for e in conns_elem:
        tag = _local(e.tag)
        c = child_map(e)
        common = dict(id=e.get('id'), from_node=e.get('from'),
                      to_node=e.get('to'))
        if tag == 'pipe':
            net.pipes.append(Pipe(
                length=_value(c['length']),
                diameter=_value(c['diameter']),
                roughness=_value(c['roughness']),
                flow_min=_flow_to_kg_per_s(c.get('flowMin'), net.norm_density)
                if 'flowMin' in c else -np.inf,
                flow_max=_flow_to_kg_per_s(c.get('flowMax'), net.norm_density)
                if 'flowMax' in c else np.inf,
                **common))
        elif tag == 'valve':
            net.valves.append(Valve(
                flow_min=_flow_to_kg_per_s(c.get('flowMin'),
                                           net.norm_density)
                if 'flowMin' in c else -np.inf,
                flow_max=_flow_to_kg_per_s(c.get('flowMax'),
                                           net.norm_density)
                if 'flowMax' in c else np.inf,
                pressure_diff_max=_value(c.get('pressureDifferentialMax'),
                                         np.inf),
                **common))
        elif tag == 'compressorStation':
            net.compressors.append(CompressorStation(
                flow_min=_flow_to_kg_per_s(c.get('flowMin'),
                                           net.norm_density)
                if 'flowMin' in c else 0.,
                flow_max=_flow_to_kg_per_s(c.get('flowMax'),
                                           net.norm_density)
                if 'flowMax' in c else np.inf,
                pressure_in_min=_value(c.get('pressureInMin'), 0.),
                pressure_out_max=_value(c.get('pressureOutMax'), np.inf),
                **common))
        # other elements (resistor, controlValve, ...) are not needed for
        # the GasLib-11/40 prototypes and are skipped

    return net


@dataclass
class BoundaryConditions:
    '''Transient boundary data from a TRR154 .bcd file. For each node id, a
    tuple (times [s], values) where values are pressures [Pa] for entries or
    mass flows [kg/s] for exits (negative = withdrawal).'''
    network: str
    t_start: float
    t_end: float
    speed_of_sound: float
    pressure: Dict[str, Tuple[np.ndarray, np.ndarray]] = \
        field(default_factory=dict)
    massflow: Dict[str, Tuple[np.ndarray, np.ndarray]] = \
        field(default_factory=dict)

    def demand(self, node_id: str, t: np.ndarray) -> np.ndarray:
        'Withdrawal (positive, kg/s) at a sink, interpolated to grid t [s].'
        times, values = self.massflow[node_id]
        return -np.interp(t, times, values)

    def pressure_at(self, node_id: str, t: np.ndarray) -> np.ndarray:
        'Boundary pressure [Pa] at an entry, interpolated to grid t [s].'
        times, values = self.pressure[node_id]
        return np.interp(t, times, values)


def parse_bcd(path) -> BoundaryConditions:
    'Parse a TRR154 .bcd boundary condition file (SI units).'
    root = ET.parse(path).getroot()
    meta = {}
    bc = None
    for child in root:
        tag = _local(child.tag)
        if tag == 'metadata':
            for e in child:
                etag = _local(e.tag)
                if etag == 'network':
                    meta['network'] = e.text
                elif etag == 'timeInterval':
                    meta['t_start'] = float(e.get('start'))
                    meta['t_end'] = float(e.get('end'))
                elif etag == 'speedOfSound':
                    meta['speed_of_sound'] = _value(e)
            bc = BoundaryConditions(
                network=meta.get('network', ''),
                t_start=meta.get('t_start', 0.),
                t_end=meta.get('t_end', 0.),
                speed_of_sound=meta.get('speed_of_sound', 340.))
        elif tag == 'nodes':
            for node in child:
                nid = node.get('id')
                times_p, vals_p, times_q, vals_q = [], [], [], []
                for nd in node:
                    t = p = q = None
                    for e in nd:
                        etag = _local(e.tag)
                        if etag == 'time':
                            t = _value(e)
                        elif etag == 'pressure':
                            p = _value(e)
                        elif etag == 'massflow':
                            q = _value(e)
                    if p is not None:
                        times_p.append(t)
                        vals_p.append(p)
                    if q is not None:
                        times_q.append(t)
                        vals_q.append(q)
                if vals_p:
                    bc.pressure[nid] = (np.array(times_p), np.array(vals_p))
                if vals_q:
                    bc.massflow[nid] = (np.array(times_q), np.array(vals_q))
    return bc


@dataclass
class NetworkState:
    '''Initial network state from a TRR154 .state file: nodal (massflow [kg/s],
    pressure [Pa]) and per-pipe spatial profiles (space [m], massflow [kg/s],
    pressure [Pa]).'''
    network: str
    speed_of_sound: float
    node_massflow: Dict[str, float] = field(default_factory=dict)
    node_pressure: Dict[str, float] = field(default_factory=dict)
    # pipe id -> (space, massflow, pressure) arrays
    pipe_profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = \
        field(default_factory=dict)
    compressor_massflow: Dict[str, float] = field(default_factory=dict)
    compressor_pressure_diff: Dict[str, float] = field(default_factory=dict)


def parse_state(path) -> NetworkState:
    'Parse a TRR154 .state file (SI units).'
    root = ET.parse(path).getroot()
    state = NetworkState(network='', speed_of_sound=340.)
    for child in root:
        tag = _local(child.tag)
        if tag == 'metadata':
            for e in child:
                etag = _local(e.tag)
                if etag == 'network':
                    state.network = e.text
                elif etag == 'speedOfSound':
                    state.speed_of_sound = _value(e)
        elif tag == 'nodes':
            for node in child:
                nid = node.get('id')
                for nd in node:
                    for e in nd:
                        etag = _local(e.tag)
                        if etag == 'massflow':
                            state.node_massflow[nid] = _value(e)
                        elif etag == 'pressure':
                            state.node_pressure[nid] = _value(e)
        elif tag == 'edges':
            for edge in child:
                eid = edge.get('id')
                etype = edge.get('type')
                if etype == 'pipe':
                    xs, qs, ps = [], [], []
                    for ed in edge:
                        x = q = p = None
                        for e in ed:
                            etag = _local(e.tag)
                            if etag == 'space':
                                x = _value(e)
                            elif etag == 'massflow':
                                q = _value(e)
                            elif etag == 'pressure':
                                p = _value(e)
                        xs.append(x)
                        qs.append(q)
                        ps.append(p)
                    state.pipe_profiles[eid] = (np.array(xs), np.array(qs),
                                                np.array(ps))
                elif etype == 'compressorStation':
                    for ed in edge:
                        for e in ed:
                            etag = _local(e.tag)
                            if etag == 'massflow':
                                state.compressor_massflow[eid] = _value(e)
                            elif etag == 'pressureDifference':
                                state.compressor_pressure_diff[eid] = \
                                    _value(e)
    return state


def validate(net: GasNetwork,
             bc: Optional[BoundaryConditions] = None) -> List[str]:
    '''Pre-flight data validation: diagnose the data failures that would
    otherwise surface as solver errors deep inside a solve (0*inf = NaN in
    the NLP, KeyError at model build, Ipopt restoration failure on
    infeasible demand). Returns a list of human-readable findings; an empty
    list means the data passed all checks.'''
    findings = []

    # switching-element bounds enter POC big-M constraints and must be finite
    for valve in net.valves:
        if not np.isfinite(valve.flow_max):
            findings.append('valve {}: flowMax missing (big-M constraint '
                            'would be 0*inf)'.format(valve.id))
    for comp in net.compressors:
        if not np.isfinite(comp.pressure_out_max):
            findings.append('compressor {}: pressureOutMax missing (big-M '
                            'constraint would be 0*inf)'.format(comp.id))
    for node in net.nodes.values():
        if not np.isfinite(node.pressure_max):
            findings.append('node {}: pressureMax missing (pressure span '
                            'used as big-M would be inf)'.format(node.id))

    # every connection endpoint must be a known node
    elements = list(net.pipes) + list(net.valves) + list(net.compressors)
    for e in elements:
        for end in (e.from_node, e.to_node):
            if end not in net.nodes:
                findings.append('{}: endpoint {} is not a node in the '
                                'network'.format(e.id, end))

    # connectivity: every node reachable from some source (undirected)
    adj: dict[str, set[str]] = {nid: set() for nid in net.nodes}
    for e in elements:
        if e.from_node in adj and e.to_node in adj:
            adj[e.from_node].add(e.to_node)
            adj[e.to_node].add(e.from_node)
    reached = set()
    stack = [n.id for n in net.sources]
    while stack:
        nid = stack.pop()
        if nid in reached:
            continue
        reached.add(nid)
        stack.extend(adj[nid] - reached)
    for nid in set(net.nodes) - reached:
        findings.append('node {}: not connected to any source'.format(nid))

    if bc is not None:
        def canon(name):
            'GasLib-11.net, GasLib_11 etc. all refer to the same network.'
            name = name.rsplit('.net', 1)[0].lower()
            return name.replace('-', '').replace('_', '')

        if bc.network and canon(bc.network) != canon(net.name):
            findings.append('boundary data is for network "{}", not '
                            '"{}"'.format(bc.network, net.name))
        for node in net.sinks:
            if node.id not in bc.massflow:
                findings.append('sink {}: no massflow boundary data in the '
                                '.bcd (model build would raise '
                                'KeyError)'.format(node.id))
        for nid in list(bc.pressure) + list(bc.massflow):
            if nid not in net.nodes:
                findings.append('.bcd node {}: not in the network'
                                .format(nid))
        # total source capacity must cover the peak of the *simultaneous*
        # demand (summing per-sink maxima would reject valid instances
        # whose sinks peak at different times)
        capacity = sum(n.flow_max for n in net.sources)
        t_all = np.unique(np.concatenate(
            [times for times, _ in bc.massflow.values()] or [np.zeros(1)]))
        total = sum((bc.demand(n.id, t_all) for n in net.sinks
                     if n.id in bc.massflow), np.zeros_like(t_all))
        peak = float(total.max()) if len(t_all) else 0.
        if np.isfinite(capacity) and peak > capacity:
            findings.append('peak simultaneous demand {:.1f} kg/s exceeds '
                            'total source capacity {:.1f} kg/s (NLP would '
                            'be infeasible)'.format(peak, capacity))

    return findings
