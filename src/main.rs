use std::{
    cmp::Ordering,
    num::FpCategory
};

use raylib::prelude::*;

use nalgebra::{Matrix2, Vector2 as NVector2, Vector3 as NVector3};


#[derive(Debug, Clone, Copy)]
pub struct WindowSize<T>
{
    pub x: T,
    pub y: T
}

impl<T> WindowSize<T>
{
    pub fn map<U>(&self, mut f: impl FnMut(&T) -> U) -> WindowSize<U>
    {
        WindowSize{x: f(&self.x), y: f(&self.y)}
    }
}

impl WindowSize<f32>
{
    pub fn scale(&self, p: Vector2) -> Vector2
    {
        Vector2{x: self.x * p.x, y: self.y * p.y}
    }

    pub fn unscale(&self, p: Vector2) -> Vector2
    {
        Vector2{x: p.x / self.x, y: p.y / self.y}
    }

    pub fn draw_circle(
        &self,
        drawing: &mut RaylibDrawHandle,
        p: Vector2,
        size: f32,
        color: Color
    )
    {
        let p = self.scale(p);

        drawing.draw_circle(
            p.x as i32,
            p.y as i32,
            size * self.x,
            color
        );
    }

    pub fn draw_line(
        &self,
        drawing: &mut RaylibDrawHandle,
        p0: Vector2,
        p1: Vector2,
        size: f32,
        color: Color
    )
    {
        drawing.draw_line_ex(
            self.scale(p0),
            self.scale(p1),
            size * self.x,
            color
        );
    }

    pub fn draw_rectangle(
        &self,
        drawing: &mut RaylibDrawHandle,
        r: Rectangle,
        angle: f32,
        color: Color
    )
    {
        let r = self.scale_rectangle(r);
        drawing.draw_rectangle_pro(
            r,
            Vector2{x: r.width / 2.0, y: r.height / 2.0},
            // who uses degrees???
            angle.to_degrees(),
            color
        );
    }

    pub fn draw_rectangle_aligned(
        &self,
        drawing: &mut RaylibDrawHandle,
        r: Rectangle,
        color: Color
    )
    {
        drawing.draw_rectangle_rec(
            self.scale_rectangle(r),
            color
        );
    }

    fn scale_rectangle(&self, r: Rectangle) -> Rectangle
    {
        Rectangle{
            x: r.x * self.x,
            y: r.y * self.y,
            width: r.width * self.x,
            height: r.height * self.x
        }
    }

    pub fn draw_line_segment(
        &self,
        drawing: &mut RaylibDrawHandle,
        p0: Vector2,
        p1: Vector2,
        size: f32
    )
    {
        self.draw_line(drawing, p0, p1, size, Color{r: 240, g: 240, b: 240, a: 255});

        let circle_size = size * 1.25;
        let circle_color = Color{r: 200, g: 200, b: 255, a: 255};

        self.draw_circle(
            drawing,
            p0,
            circle_size,
            circle_color
        );

        self.draw_circle(
            drawing,
            p1,
            circle_size,
            circle_color
        );
    }
}

pub fn point_line_side(p: NVector2<f32>, a: NVector2<f32>, b: NVector2<f32>) -> Ordering
{
    let x = project_onto_line(p, a, b);
    if x < 0.0
    {
        Ordering::Less
    } else if x > 1.0
    {
        Ordering::Greater
    } else
    {
        Ordering::Equal
    }
}

fn project_onto_line(p: NVector2<f32>, a: NVector2<f32>, b: NVector2<f32>) -> f32
{
    let ad = b.metric_distance(&p);
    let cd = a.metric_distance(&b);
    let bd = a.metric_distance(&p);

    let cosa = (ad.powi(2) - bd.powi(2) - cd.powi(2)) / (-2.0 * bd * cd);

    cosa * bd / cd
}

pub fn point_line_distance(p: NVector2<f32>, a: NVector2<f32>, b: NVector2<f32>) -> f32
{
    let check = match point_line_side(p, a, b)
    {
        Ordering::Equal =>
        {
            let diff = b - a;

            return cross_2d(diff, a - p).abs() / diff.magnitude();
        },
        Ordering::Less => a,
        Ordering::Greater => b
    };

    p.metric_distance(&check)
}

pub fn cross_2d(a: NVector2<f32>, b: NVector2<f32>) -> f32
{
    a.x * b.y - b.x * a.y
}

pub fn cross_3d(a: NVector3<f32>, b: NVector3<f32>) -> NVector3<f32>
{
    NVector3::new(
        cross_2d(a.yz(), b.yz()),
        cross_2d(a.zx(), b.zx()),
        cross_2d(a.xy(), b.xy())
    )
}

pub fn rotate_point(p: NVector2<f32>, angle: f32) -> NVector2<f32>
{
    let (asin, acos) = angle.sin_cos();

    NVector2::new(acos * p.x + asin * p.y, -asin * p.x + acos * p.y)
}

#[derive(Debug)]
pub struct NTransform
{
    pub position: NVector2<f32>,
    pub scale: NVector2<f32>,
    pub rotation: f32
}

impl NTransform
{
    pub fn project(&self, p: NVector2<f32>) -> NVector2<f32>
    {
        rotate_point(p - self.position, self.rotation).component_div(&self.scale)
    }

    pub fn unproject(&self, p: NVector2<f32>) -> NVector2<f32>
    {
        rotate_point(p.component_mul(&self.scale), -self.rotation) + self.position
    }
}

pub fn rectangle_points(transform: &NTransform) -> [NVector2<f32>; 4]
{
    let size = transform.scale;
    let pos = transform.position;
    let rotation = transform.rotation;

    let x_shift = NVector2::new(size.x / 2.0, 0.0);
    let y_shift = NVector2::new(0.0, size.y / 2.0);

    let pos = pos.xy();

    let left_middle = pos - x_shift;
    let right_middle = pos + x_shift;

    [
        left_middle - y_shift,
        right_middle - y_shift,
        right_middle + y_shift,
        left_middle + y_shift
    ].map(|x|
    {
        rotate_point(x - pos, -rotation) + pos
    })
}

pub fn get_two_mut<T>(s: &mut [T], one: usize, two: usize) -> (&mut T, &mut T)
{
    if one > two
    {
 
        let (left, right) = s.split_at_mut(one);

        (&mut right[0], &mut left[two])
    } else
    {
        let (left, right) = s.split_at_mut(two);

        (&mut left[one], &mut right[0])
    }
}

pub const GRAVITY: NVector2<f32> = NVector2::new(0.0, 0.2);
pub const DEFAULT_RESTITUTION: f32 = 0.3;
pub const ANGULAR_LIMIT: f32 = 2.0;
pub const VELOCITY_LOW: f32 = 0.05;

struct Inertias
{
    angular: f32,
    linear: f32
}

enum WhichObject
{
    A,
    B
}

trait IteratedMoves
{
    fn inverted(self) -> Self;
}

#[derive(Debug, Clone, Copy)]
struct PenetrationMoves
{
    pub velocity_change: NVector2<f32>,
    pub angular_change: f32,
    pub angular_amount: f32
}

impl IteratedMoves for PenetrationMoves
{
    fn inverted(self) -> Self
    {
        Self{
            angular_amount: -self.angular_amount,
            ..self
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct VelocityMoves
{
    pub velocity_change: NVector2<f32>,
    pub angular_change: f32,
    pub inverted: bool
}

impl IteratedMoves for VelocityMoves
{
    fn inverted(self) -> Self
    {
        Self{
            inverted: true,
            ..self
        }
    }
}

#[derive(Debug, Clone)]
pub struct Contact
{
    pub a: usize,
    pub b: Option<usize>,
    pub point: NVector2<f32>,
    pub normal: NVector2<f32>,
    pub penetration: f32,
    pub restitution: f32
}

#[derive(Debug, Clone)]
struct AnalyzedContact
{
    pub contact: Contact,
    pub to_world: Matrix2<f32>,
    pub closing: NVector2<f32>,
    pub desired_change: f32,
    pub a_relative: NVector2<f32>,
    pub b_relative: Option<NVector2<f32>>
}

impl AnalyzedContact
{
    fn calculate_desired_change(
        &self,
        objects: &[Object],
        dt: f32
    ) -> f32
    {
        self.contact.calculate_desired_change(objects, &self.closing, dt)
    }

    fn inertias(&self, objects: &[Object], which: WhichObject) -> Inertias
    {
        let (object, contact_relative) = self.get(objects, which);

        // this always returns 0 im pretty sure lol, in 2d at least
        let angular_inertia_world = Contact::direction_apply_inertia(
            object.inertia(),
            contact_relative,
            self.contact.normal
        );

        Inertias{
            linear: object.physical.inverse_mass,
            angular: angular_inertia_world.dot(&self.contact.normal)
        }
    }

    fn get<'a>(&self, objects: &'a [Object], which: WhichObject) -> (&'a Object, NVector2<f32>)
    {
        match which
        {
            WhichObject::A =>
            {
                (&objects[self.contact.a], self.a_relative)
            },
            WhichObject::B =>
            {
                (&objects[self.contact.b.unwrap()], self.b_relative.unwrap())
            }
        }
    }

    fn get_mut<'a>(
        &self,
        objects: &'a mut [Object],
        which: WhichObject
    ) -> (&'a mut Object, NVector2<f32>)
    {
        match which
        {
            WhichObject::A =>
            {
                (&mut objects[self.contact.a], self.a_relative)
            },
            WhichObject::B =>
            {
                (&mut objects[self.contact.b.unwrap()], self.b_relative.unwrap())
            }
        }
    }

    fn apply_moves(
        &self,
        objects: &mut [Object],
        inertias: Inertias,
        inverse_inertia: f32,
        which: WhichObject
    ) -> PenetrationMoves
    {
        let penetration = match which
        {
            WhichObject::A => self.contact.penetration,
            WhichObject::B => -self.contact.penetration
        } * inverse_inertia;

        let mut angular_amount = penetration * inertias.angular;
        let mut linear_change = penetration * inertias.linear;

        let (object, contact_relative) = self.get_mut(objects, which);

        let angular_limit = ANGULAR_LIMIT * contact_relative.magnitude();

        if angular_amount.abs() > angular_limit
        {
            let pre_limit = angular_amount;

            angular_amount = angular_amount.clamp(-angular_limit, angular_limit);

            linear_change += pre_limit - angular_amount;
        }

        let velocity_change = linear_change * self.contact.normal;
        object.transform.position += velocity_change;

        let impulse_torque = cross_2d(
            contact_relative,
            self.contact.normal
        ) / object.inertia();

        let angular_change = if inertias.angular.classify() != FpCategory::Zero
        {
            let angular_change_unit = impulse_torque / inertias.angular;

            let angular_change = angular_change_unit * angular_amount;
            object.transform.rotation += angular_change;

            angular_change
        } else
        {
            0.0
        };

        PenetrationMoves{
            velocity_change,
            angular_change,
            angular_amount
        }
    }

    fn resolve_penetration(
        &self,
        objects: &mut [Object]
    ) -> (PenetrationMoves, Option<PenetrationMoves>)
    {
        let inertias@Inertias{angular, linear} = self.inertias(objects, WhichObject::A);
        let mut total_inertia = angular + linear;

        let b_inertias = self.contact.b.map(|_|
        {
            self.inertias(objects, WhichObject::B)
        });

        if let Some(Inertias{angular: b_angular, linear: b_linear}) = b_inertias
        {
            total_inertia += b_angular + b_linear;
        }

        let inverse_inertia = total_inertia.recip();

        let a_moves = self.apply_moves(objects, inertias, inverse_inertia, WhichObject::A);

        let b_moves = b_inertias.map(|b_inertias|
        {
            self.apply_moves(
                objects,
                b_inertias, 
                inverse_inertia,
                WhichObject::B
            )
        });

        (a_moves, b_moves)
    }

    fn velocity_change(
        &self,
        objects: &[Object],
        which: WhichObject
    ) -> f32
    {
        let (object, contact_relative) = self.get(objects, which);

        let velocity_change_world = Contact::direction_apply_inertia(
            object.inertia(),
            contact_relative,
            self.contact.normal
        );

        velocity_change_world.dot(&self.contact.normal) + object.physical.inverse_mass
    }

    fn apply_impulse(
        &self,
        objects: &mut [Object],
        impulse: NVector2<f32>,
        which: WhichObject
    ) -> VelocityMoves
    {
        let (object, contact_relative) = self.get_mut(objects, which);

        let impulse_torque = cross_2d(contact_relative, impulse);

        let angular_change = impulse_torque / object.inertia();
        let velocity_change = impulse * object.physical.inverse_mass;

        object.physical.velocity += velocity_change;
        object.physical.angular_velocity += angular_change;

        VelocityMoves{
            angular_change,
            velocity_change,
            inverted: false
        }
    }

    fn resolve_velocity(
        &self,
        objects: &mut [Object]
    ) -> (VelocityMoves, Option<VelocityMoves>)
    {
        let mut velocity_change = self.velocity_change(objects, WhichObject::A);

        if let Some(_) = self.contact.b
        {
            let b_velocity_change = self.velocity_change(objects, WhichObject::B);

            velocity_change += b_velocity_change;
        }

        let impulse = self.to_world * NVector2::new(
            self.desired_change / velocity_change,
            0.0
        );

        let a_moves = self.apply_impulse(objects, impulse, WhichObject::A);

        let b_moves = self.contact.b.map(|_|
        {
            self.apply_impulse(objects, -impulse, WhichObject::B)
        });

        (a_moves, b_moves)
    }
}

impl Contact
{
    pub fn to_world_matrix(&self) -> Matrix2<f32>
    {
        let cosa = self.normal.x;
        let msina = self.normal.y;

        Matrix2::new(
            cosa, -msina,
            msina, cosa
        )
    }

    fn direction_apply_inertia(
        inertia: f32,
        direction: NVector2<f32>,
        normal: NVector2<f32>
    ) -> NVector2<f32>
    {
        let direction_3d = NVector3::new(direction.x, direction.y, 0.0);

        let angular_inertia = cross_3d(
            direction_3d,
            NVector3::new(normal.x, normal.y, 0.0)
        ) / inertia;

        cross_3d(
            angular_inertia,
            direction_3d
        ).xy()
    }

    fn velocity_from_angular(angular: f32, contact_local: &NVector2<f32>) -> NVector2<f32>
    {
        cross_3d(
            NVector3::new(0.0, 0.0, angular),
            NVector3::new(contact_local.x, contact_local.y, 0.0)
        ).xy()
    }

    fn velocity_closing(
        object: &Object,
        to_world: &Matrix2<f32>,
        contact_relative: &NVector2<f32>
    ) -> NVector2<f32>
    {
        let closing_world = Self::velocity_from_angular(
            object.physical.angular_velocity,
            contact_relative
        ) + object.physical.velocity;

        to_world.transpose() * closing_world
    }

    fn calculate_desired_change(
        &self,
        objects: &[Object],
        closing: &NVector2<f32>,
        dt: f32
    ) -> f32
    {
        let mut acceleration_velocity = (objects[self.a].physical.last_acceleration * dt)
            .dot(&self.normal);

        if let Some(b) = self.b
        {
            acceleration_velocity -= (objects[b].physical.last_acceleration * dt).dot(&self.normal);
        }

        let restitution = if closing.x.abs() < VELOCITY_LOW
        {
            0.0
        } else
        {
            self.restitution
        };

        -closing.x - restitution * (closing.x - acceleration_velocity)
    }

    fn analyze(self, objects: &[Object], dt: f32) -> AnalyzedContact
    {
        let to_world = self.to_world_matrix();

        let a_relative = self.point - objects[self.a].transform.position;
        let b_relative = self.b.map(|b| self.point - objects[b].transform.position);

        let mut closing = Self::velocity_closing(&objects[self.a], &to_world, &a_relative);
        if let Some(b) = self.b
        {
            closing -= Self::velocity_closing(
                &objects[b],
                &to_world,
                b_relative.as_ref().unwrap()
            );
        }

        let desired_change = self.calculate_desired_change(objects, &closing, dt);

        AnalyzedContact{
            to_world,
            closing,
            desired_change,
            a_relative,
            b_relative,
            contact: self
        }
    }
}

pub struct ContactResolver;

impl ContactResolver
{
    pub fn new() -> Self
    {
        Self
    }

    fn update_iterated<Moves: IteratedMoves + Copy>(
        &mut self,
        objects: &[Object],
        contacts: &mut [AnalyzedContact],
        moves: (Moves, Option<Moves>),
        bodies: (usize, Option<usize>),
        mut handle: impl FnMut(&[Object], &mut AnalyzedContact, Moves, NVector2<f32>)
    )
    {
        let (a_move, b_move) = moves;
        let (a_id, b_id) = bodies;

        contacts.iter_mut().filter_map(|x|
        {
            let relative = |id: usize|
            {
                x.contact.point - objects[id].transform.position
            };

            if x.contact.a == a_id
            {
                Some((a_move.inverted(), relative(x.contact.a)))
            } else if Some(x.contact.a) == b_id
            {
                Some((b_move.unwrap().inverted(), relative(x.contact.a)))
            } else if x.contact.b == Some(a_id)
            {
                Some((a_move, relative(x.contact.b.unwrap())))
            } else if x.contact.b.is_some() && x.contact.b == b_id
            {
                Some((b_move.unwrap(), relative(x.contact.b.unwrap())))
            } else
            {
                None
            }.map(|b| (x, b))
        }).for_each(|(contact, (move_info, rest_info))|
        {
            handle(objects, contact, move_info, rest_info);
        });
    }

    fn resolve_iterative<Moves: IteratedMoves + Copy>(
        &mut self,
        objects: &mut [Object],
        contacts: &mut [AnalyzedContact],
        iterations: usize,
        compare: impl Fn(&AnalyzedContact) -> f32,
        mut resolver: impl FnMut(&mut [Object], &mut AnalyzedContact) -> (Moves, Option<Moves>),
        mut updater: impl FnMut(&[Object], &mut AnalyzedContact, Moves, NVector2<f32>)
    )
    {
        for _ in 0..iterations
        {
            if let Some(contact) = contacts.iter_mut().map(|contact|
            {
                (compare(contact), contact)
            }).max_by(|(a, _), (b, _)|
            {
                a.partial_cmp(b).unwrap_or(Ordering::Less)
            }).map(|(_, x)| x).filter(|contact| compare(contact) > 0.0)
            {
                let moves = resolver(objects, contact);
                let bodies = (contact.contact.a, contact.contact.b);

                debug_assert!(moves.1.is_some() == contact.contact.b.is_some());

                self.update_iterated::<Moves>(
                    objects,
                    contacts,
                    moves,
                    bodies,
                    &mut updater
                );
            } else
            {
                break;
            }
        }
    }

    pub fn resolve(
        &mut self,
        objects: &mut [Object],
        contacts: &mut Vec<Contact>,
        dt: f32
    )
    {
        let mut analyzed_contacts: Vec<_> = contacts.iter().cloned().map(|contact|
        {
            contact.analyze(objects, dt)
        }).collect();

        self.resolve_iterative(
            objects,
            &mut analyzed_contacts,
            4,
            |contact| contact.contact.penetration,
            |objects, contact| contact.resolve_penetration(objects),
            |_obejcts, contact, move_info, contact_relative|
            {
                let contact_change = Contact::velocity_from_angular(
                    move_info.angular_change,
                    &contact_relative
                ) + move_info.velocity_change;

                contact.contact.penetration -= contact_change.dot(&contact.contact.normal);
            }
        );

        self.resolve_iterative(
            objects,
            &mut analyzed_contacts,
            4,
            |contact| contact.desired_change,
            |objects, contact| contact.resolve_velocity(objects),
            |objects, contact, move_info, contact_relative|
            {
                let contact_change = Contact::velocity_from_angular(
                    move_info.angular_change,
                    &contact_relative
                ) + move_info.velocity_change;

                let change = contact.to_world.transpose() * contact_change;

                if move_info.inverted
                {
                    contact.closing += change;
                } else
                {
                    contact.closing -= change;
                }

                contact.desired_change = contact.calculate_desired_change(objects, dt);
            }
        );

        contacts.clear();
    }
}

pub struct PhysicalProperties
{
    pub inverse_mass: f32,
    pub damping: f32,
    pub angular_damping: f32
}

pub struct Physical
{
    inverse_mass: f32,
    angular_damping: f32,
    torgue: f32,
    angular_velocity: f32,
    angular_acceleration: f32,
    damping: f32,
    force: NVector2<f32>,
    velocity: NVector2<f32>,
    acceleration: NVector2<f32>,
    last_acceleration: NVector2<f32>
}

impl Physical
{
    pub fn new(props: PhysicalProperties) -> Self
    {
        Self{
            inverse_mass: props.inverse_mass,
            angular_damping: props.angular_damping,
            torgue: 0.0,
            angular_velocity: 0.0,
            angular_acceleration: 0.0,
            damping: props.damping,
            force: NVector2::zeros(),
            velocity: NVector2::zeros(),
            acceleration: NVector2::zeros(),
            last_acceleration: NVector2::zeros()
        }
    }

    pub fn update(&mut self, transform: &mut NTransform, inertia: f32, dt: f32)
    {
        transform.position += self.velocity * dt;
        transform.rotation += self.angular_velocity * dt;

        self.last_acceleration = self.acceleration + self.force * self.inverse_mass;

        self.velocity += self.last_acceleration * dt;
        self.velocity *= self.damping.powf(dt);

        self.force = NVector2::zeros();

        if self.inverse_mass != 0.0
        {
            let angular_acceleration = self.angular_acceleration + self.torgue / inertia;

            self.angular_velocity += angular_acceleration * dt;
            self.angular_velocity *= self.angular_damping.powf(dt);

            self.torgue = 0.0;
        }
    }

    pub fn set_acceleration(&mut self, acceleration: NVector2<f32>)
    {
        self.acceleration = acceleration;
    }

    pub fn add_force(&mut self, force: NVector2<f32>)
    {
        self.force += force;
    }

    fn add_force_at_point(&mut self, force: NVector2<f32>, point: NVector2<f32>)
    {
        self.force += force;

        self.torgue += cross_2d(point, force);
    }
}

pub fn spring_one_way(
    a: &mut Object,
    point: NVector2<f32>,
    b: NVector2<f32>,
    strength: f32,
    length: f32
)
{
    let distance = a.transform.position - b;

    let magnitude = distance.magnitude();
    match magnitude.classify()
    {
        FpCategory::Nan
        | FpCategory::Infinite
        | FpCategory::Zero => return,
        _ => ()
    }

    let amount = magnitude - length;

    a.add_force_at(distance.normalize() * amount * -strength, point);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape
{
    Rectangle,
    Circle
}

impl Shape
{
    fn inside_rectangle(
        p: NVector2<f32>,
        a: NVector2<f32>,
        b: NVector2<f32>,
        d: NVector2<f32>
    ) -> bool
    {
        let inside = move |a, b|
        {
            point_line_side(p, a, b) == Ordering::Equal
        };

        inside(a, b) && inside(a, d)
    }

    fn point_inside(
        &self,
        transform: &NTransform,
        point: NVector2<f32>
    ) -> bool
    {
        match self
        {
            Self::Rectangle =>
            {
                let [a, b, _c, d] = rectangle_points(transform);

                Self::inside_rectangle(point, a, b, d)
            },
            Self::Circle =>
            {
                transform.position.metric_distance(&point) < transform.scale.max() / 2.0
            }
        }
    }
}

#[derive(Debug)]
struct TransformMatrix<'a>
{
    pub transform: &'a NTransform,
    pub rotation_matrix: Matrix2<f32>
}

impl<'a> From<&'a NTransform> for TransformMatrix<'a>
{
    fn from(transform: &'a NTransform) -> TransformMatrix<'a>
    {
        let rotation = -transform.rotation;

        Self{
            transform,
            rotation_matrix: Matrix2::new(
                rotation.cos(), rotation.sin(),
                -rotation.sin(), rotation.cos()
            )
        }
    }
}

impl TransformMatrix<'_>
{
    fn rectangle_on_axis(&self, axis: &NVector2<f32>) -> f32
    {
        self.transform.scale.iter().zip(self.rotation_matrix.column_iter()).map(|(scale, column)|
        {
            (scale / 2.0) * axis.dot(&column).abs()
        }).sum()
    }

    pub fn overlapping_axis(
        &self,
        other: &Self,
        axis: &NVector2<f32>
    ) -> bool
    {
        let this_projected = self.rectangle_on_axis(axis);
        let other_projected = other.rectangle_on_axis(axis);

        let diff = other.transform.position - self.transform.position;

        let axis_distance = diff.dot(axis).abs();

        axis_distance < this_projected + other_projected
    }

    fn penetration_rectangle_point(
        &self,
        other: &NVector2<f32>,
        this_id: usize,
        other_id: usize
    ) -> Option<Contact>
    {
        let circle_projected = rotate_point(
            other - self.transform.position,
            self.transform.rotation
        );

        let distance = self.transform.scale / 2.0 - circle_projected.abs();

        if distance.iter().any(|x| *x < 0.0)
        {
            return None;
        }

        let mut normals = self.rotation_matrix.column_iter().zip(circle_projected.into_iter())
            .map(|(axis, pos)|
            {
                axis * *pos
            });

        let mut shallowest = distance.x;
        let mut normal = normals.next().unwrap();

        macro_rules! check_axis
        {
            ($field:ident) =>
            {
                let next_normal = normals.next().unwrap();
                if distance.$field < shallowest
                {
                    shallowest = distance.$field;
                    normal = next_normal;
                }
            }
        }

        check_axis!(y);

        Some(Contact{
            a: this_id,
            b: Some(other_id),
            point: *other,
            penetration: shallowest,
            normal: -normal.normalize(),
            restitution: DEFAULT_RESTITUTION
        })
    }

    pub fn intersecting_rectangle_rectangle(
        &self,
        other: &Self
    ) -> bool
    {
        let with_axis = |axis: NVector2<f32>| -> bool
        {
            self.overlapping_axis(&other, &axis)
        };

        let dims = 2;
        (0..dims).all(|i| with_axis(self.rotation_matrix.column(i).into()))
            && (0..dims).all(|i| with_axis(other.rotation_matrix.column(i).into()))

        // cross products for 3d
        /*(0..dims).all(|i| with_axis(this.matrix.column(i).into()))
            && (0..dims).all(|i| with_axis(other.matrix.column(i).into()))
            && (0..dims).all(|i|
                {
                    (0..dims).all(move |j|
                    {
                        with_axis(this.matrix.column(i).cross(&other.matrix.column(j)))
                    })
                })*/
    }

    pub fn rectangle_rectangle_contact(
        &self,
        other: &Self,
        this_id: usize,
        other_id: usize
    ) -> Option<Contact>
    {
        // in 3d find contacts between the edges

        rectangle_points(&self.transform).into_iter().filter_map(|point|
        {
            other.penetration_rectangle_point(&point, other_id, this_id)
        }).chain(rectangle_points(&other.transform).into_iter().filter_map(|point|
        {
            self.penetration_rectangle_point(&point, this_id, other_id)
        })).max_by(|a, b| a.penetration.partial_cmp(&b.penetration).unwrap())
    }
}

pub struct Object
{
    transform: NTransform,
    shape: Shape,
    pub physical: Physical
}

impl Object
{
    pub fn new(
        transform: NTransform,
        shape: Shape,
        physical: PhysicalProperties
    ) -> Self
    {
        let physical = Physical::new(physical);

        Self{transform, shape, physical}
    }

    pub fn point_inside(&self, point: NVector2<f32>) -> bool
    {
        self.shape.point_inside(&self.transform, point)
    }

    pub fn inertia(&self) -> f32
    {
        match self.shape
        {
            Shape::Rectangle =>
            {
                let w = self.transform.scale.x;
                let h = self.transform.scale.y;

                (1.0/12.0) * self.physical.inverse_mass.recip() * (w.powi(2) + h.powi(2))
            },
            Shape::Circle =>
            {
                (1.0/2.0) * self.physical.inverse_mass.recip() * self.transform.scale.max().powi(2)
            }
        }
    }

    pub fn update(&mut self, dt: f32)
    {
        // self.physical.acceleration = GRAVITY;

        let inertia = self.inertia();
        self.physical.update(&mut self.transform, inertia, dt);
    }

    fn collide_circle_circle(
        &self,
        other: &Self,
        contacts: &mut Vec<Contact>,
        this_id: usize,
        other_id: usize
    )
    {
        let this_radius = self.transform.scale.max() / 2.0;
        let other_radius = other.transform.scale.max() / 2.0;

        let diff = other.transform.position - self.transform.position;
        let distance = diff.magnitude();

        if (distance - this_radius - other_radius) >= 0.0
        {
            return;
        }

        let normal = diff / distance;

        let contact = Contact{
            a: this_id,
            b: Some(other_id),
            point: self.transform.position + normal * this_radius, // is this correct?
            penetration: this_radius + other_radius - distance,
            normal: -normal,
            restitution: DEFAULT_RESTITUTION
        };

        contacts.push(contact);
    }

    fn collide_rectangle_circle(
        &self,
        other: &Self,
        contacts: &mut Vec<Contact>,
        this_id: usize,
        other_id: usize
    )
    {
        let circle_projected = rotate_point(
            other.transform.position - self.transform.position,
            self.transform.rotation
        );

        let closest_point_local = (self.transform.scale / 2.0).zip_map(&circle_projected, |a, b|
        {
            b.clamp(-a, a)
        });

        let diff = circle_projected - closest_point_local;
        let squared_distance = diff.x.powi(2) + diff.y.powi(2);

        let radius = other.transform.scale.max() / 2.0;
        if squared_distance > radius.powi(2)
        {
            return;
        }

        let closest_point =  rotate_point(
            closest_point_local,
            -self.transform.rotation
        ) + self.transform.position;

        let normal = -(other.transform.position - closest_point).try_normalize(0.0001)
            .unwrap_or_else(||
            {
                -(self.transform.position - closest_point).try_normalize(0.0001).unwrap_or_else(||
                {
                    NVector2::new(1.0, 0.0)
                })
            });

        let contact = Contact{
            a: this_id,
            b: Some(other_id),
            point: closest_point,
            penetration: radius - squared_distance.sqrt(),
            normal,
            restitution: DEFAULT_RESTITUTION
        };

        contacts.push(contact);
    }

    fn collide_rectangle_rectangle(
        &self,
        other: &Self,
        contacts: &mut Vec<Contact>,
        this_id: usize,
        other_id: usize
    )
    {
        let this = TransformMatrix::from(&self.transform);
        let other = TransformMatrix::from(&other.transform);

        if !this.intersecting_rectangle_rectangle(&other)
        {
            return;
        }

        contacts.extend(this.rectangle_rectangle_contact(&other, this_id, other_id));
    }

    pub fn collide(
        &self,
        other: &Self,
        contacts: &mut Vec<Contact>,
        this_id: usize,
        other_id: usize
    )
    {
        match (self.shape, other.shape)
        {
            (Shape::Circle, Shape::Circle) => self.collide_circle_circle(other, contacts, this_id, other_id),
            (Shape::Rectangle, Shape::Circle) => self.collide_rectangle_circle(other, contacts, this_id, other_id),
            (Shape::Circle, Shape::Rectangle) => other.collide_rectangle_circle(self, contacts, other_id, this_id),
            (Shape::Rectangle, Shape::Rectangle) => self.collide_rectangle_rectangle(other, contacts, this_id, other_id)
        }
    }

    pub fn collide_plane(
        &self,
        contacts: &mut Vec<Contact>,
        normal: NVector2<f32>,
        offset: f32,
        this_id: usize
    )
    {
        match self.shape
        {
            Shape::Circle =>
            {
                let distance = self.transform.position.dot(&normal) - offset;

                let radius = self.transform.scale.max() / 2.0;
                let total_distance = distance - radius;

                if total_distance >= 0.0
                {
                    return;
                }

                let contact = Contact{
                    a: this_id,
                    b: None,
                    point: self.transform.position - normal * distance,
                    penetration: -total_distance,
                    normal,
                    restitution: DEFAULT_RESTITUTION
                };

                contacts.push(contact);
            },
            Shape::Rectangle =>
            {
                rectangle_points(&self.transform).into_iter().for_each(|point|
                {
                    let distance = point.dot(&normal) - offset;

                    if distance >= 0.0
                    {
                        return;
                    }

                    let contact = Contact{
                        a: this_id,
                        b: None,
                        point: point - normal * distance,
                        penetration: -distance,
                        normal,
                        restitution: DEFAULT_RESTITUTION
                    };

                    contacts.push(contact);
                });
            }
        }
    }

    pub fn add_force_at(&mut self, force: NVector2<f32>, point: NVector2<f32>)
    {
        let point = self.transform.unproject(point) - self.transform.position;
        self.physical.add_force_at_point(force, point)
    }

    pub fn draw(&self, size: WindowSize<f32>, drawing: &mut RaylibDrawHandle)
    {
        let color = Color{r: 100, g: 100, b: 170, a: 255};
        match self.shape
        {
            Shape::Rectangle =>
            {
                size.draw_rectangle(
                    drawing,
                    rectangle_from(self.transform.position, self.transform.scale),
                    self.transform.rotation,
                    color
                );
            },
            Shape::Circle =>
            {
                size.draw_circle(
                    drawing,
                    uncvt(self.transform.position),
                    self.transform.scale.max() / 2.0,
                    color
                );
            }
        }
    }
}

fn rectangle_from(pos: NVector2<f32>, scale: NVector2<f32>) -> Rectangle
{
    Rectangle{
        x: pos.x,
        y: pos.y,
        width: scale.x,
        height: scale.y
    }
}

fn cvt(v: Vector2) -> NVector2<f32>
{
    NVector2::new(v.x, v.y)
}

fn uncvt(v: NVector2<f32>) -> Vector2
{
    Vector2{x: v.x, y: v.y}
}

/*#[link(name = "floathelper")]
extern "C"
{
    fn float_excepts();
}*/

fn main()
{
    // unsafe{ float_excepts(); }
    let size: WindowSize<i32> = WindowSize{x: 640, y: 640};
    let (mut raylib, this_thread) = raylib::init()
        .log_level(TraceLogLevel::LOG_WARNING)
        .size(size.x, size.y)
        .title("collision tests")
        .build();

    let size = size.map(|x| *x as f32);

    let mut objects = vec![];

    let rectangle_aspect = 1.5;

    let mut holding_left = false;
    let mut mouse_object_size = 0.0;
    let grow_speed = 0.1;

    let mouse_scale = |size|
    {
        NVector2::repeat(size).component_mul(&NVector2::new(rectangle_aspect, 1.0))
    };

    let mut current_shape = Shape::Rectangle;
    let mut held_object = None;

    let mut contact_resolver = ContactResolver::new();

    #[derive(PartialEq, Eq)]
    enum FrameType
    {
        Integration,
        Collision
    }

    let mut frame_counter = 0;
    let mut frame_type = FrameType::Integration;

    let long_wait_time = 1.0;
    let dt = 1.0 / 60.0;
    while !raylib.window_should_close()
    {
        let mut drawing = raylib.begin_drawing(&this_thread);

        drawing.clear_background(Color{r: 25, g: 25, b: 50, a: 255});

        let mouse_position = size.unscale(drawing.get_mouse_position());

        if drawing.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT)
        {
            holding_left = true;
        } else if drawing.is_mouse_button_released(MouseButton::MOUSE_BUTTON_LEFT)
        {
            let transform = NTransform{
                position: cvt(mouse_position),
                scale: match current_shape
                {
                    Shape::Rectangle => mouse_scale(mouse_object_size),
                    Shape::Circle => NVector2::repeat(mouse_object_size * 2.0)
                },
                rotation: 0.0
            };

            let mass = match current_shape
            {
                Shape::Rectangle => mouse_scale(mouse_object_size).product(),
                Shape::Circle => mouse_object_size.powi(2) * std::f32::consts::PI
            } * 55.0;

            let physical = PhysicalProperties{
                inverse_mass: mass.recip(),
                damping: 0.9,
                angular_damping: 0.9
            };

            objects.push(Object::new(transform, current_shape, physical));

            holding_left = false;
            mouse_object_size = 0.0;
        }

        let long_wait = drawing.is_key_down(KeyboardKey::KEY_S);

        /*
        let speed = 0.05;
        let rotate_speed = 0.5;
        let scale_speed = 0.05;

        let control_id = 1;

        let mut try_move = |direction|
        {
            if let Some(object) = objects.get_mut(control_id)
            {
                object.transform.position += direction * speed * dt;
            }
        };

        if drawing.is_key_down(KeyboardKey::KEY_W) { try_move(NVector2::new(0.0, -1.0)); }
        if drawing.is_key_down(KeyboardKey::KEY_S) { try_move(NVector2::new(0.0, 1.0)); }
        if drawing.is_key_down(KeyboardKey::KEY_A) { try_move(NVector2::new(-1.0, 0.0)); }
        if drawing.is_key_down(KeyboardKey::KEY_D) { try_move(NVector2::new(1.0, 0.0)); }

        if drawing.is_key_down(KeyboardKey::KEY_Z)
        {
            if let Some(object) = objects.get_mut(control_id)
            {
                object.transform.rotation -= rotate_speed * dt;
            }
        }

        if drawing.is_key_down(KeyboardKey::KEY_X)
        {
            if let Some(object) = objects.get_mut(control_id)
            {
                object.transform.rotation += rotate_speed * dt;
            }
        }

        if drawing.is_key_down(KeyboardKey::KEY_C)
        {
            if let Some(object) = objects.get_mut(control_id)
            {
                object.transform.scale -= NVector2::repeat(scale_speed * dt);
            }
        }

        if drawing.is_key_down(KeyboardKey::KEY_V)
        {
            if let Some(object) = objects.get_mut(control_id)
            {
                object.transform.scale += NVector2::repeat(scale_speed * dt);
            }
        }*/

        if drawing.is_key_pressed(KeyboardKey::KEY_F)
        {
            current_shape = match current_shape
            {
                Shape::Rectangle => Shape::Circle,
                Shape::Circle => Shape::Rectangle
            };
        }

        if drawing.is_key_pressed(KeyboardKey::KEY_E)
        {
            held_object = objects.iter()
                .position(|x| x.point_inside(cvt(mouse_position)))
                .map(|x|
                {
                    (x, objects[x].transform.project(cvt(mouse_position)))
                });
        } else if drawing.is_key_released(KeyboardKey::KEY_E)
        {
            held_object.take();
        }

        let objects_len = objects.len();

        if !long_wait || (frame_counter == 0 && frame_type == FrameType::Integration)
        {
            if let Some((id, point)) = held_object
            {
                spring_one_way(&mut objects[id], point, cvt(mouse_position), 3.0, 0.0);
            }

            if holding_left
            {
                mouse_object_size += grow_speed * dt;

                let color = Color{r: 100, g: 50, b: 50, a: 255};
                match current_shape
                {
                    Shape::Rectangle =>
                    {
                        let scale = mouse_scale(mouse_object_size);

                        let rpos = cvt(mouse_position) - scale / 2.0;

                        size.draw_rectangle_aligned(
                            &mut drawing,
                            rectangle_from(rpos, scale),
                            color
                        );
                    },
                    Shape::Circle =>
                    {
                        size.draw_circle(
                            &mut drawing,
                            mouse_position,
                            mouse_object_size,
                            color
                        );
                    }
                }
            }

            for id in 0..objects_len
            {
                objects[id].update(dt);
            }
        }

        for id in 0..objects_len
        {
            objects[id].draw(size, &mut drawing);
        }

        let mut contacts = Vec::new();
        for id in 0..objects_len
        {
            {
                let object = &mut objects[id];
                object.collide_plane(&mut contacts, NVector2::new(0.0, -1.0), -1.0, id);
                object.collide_plane(&mut contacts, NVector2::new(0.0, 1.0), 0.0, id);
                object.collide_plane(&mut contacts, NVector2::new(-1.0, 0.0), -1.0, id);
                object.collide_plane(&mut contacts, NVector2::new(1.0, 0.0), 0.0, id);
            }

            let mut pairs_fn = |a_id, b_id|
            {
                let (a, b) = get_two_mut(&mut objects, a_id, b_id);
                a.collide(b, &mut contacts, a_id, b_id);
            };

            {
                let mut colliders = 0..objects_len;

                // calls the function for each unique combination (excluding (entities, entities) pairs)
                colliders.clone().for_each(|a|
                {
                    colliders.by_ref().next();
                    colliders.clone().for_each(|b| pairs_fn(a, b));
                });
            }
        }

        contacts.iter().for_each(|contact|
        {
            size.draw_circle(
                &mut drawing,
                uncvt(contact.point),
                contact.penetration,
                Color{r: 255, g: 50, b: 50, a: 255}
            );

            size.draw_line(
                &mut drawing,
                uncvt(contact.point),
                uncvt(contact.point + contact.normal / size.x * 15.0),
                0.01,
                Color{r: 255, g: 100, b: 100, a: 255}
            );
        });

        if !long_wait || (frame_counter == 0 && frame_type == FrameType::Collision)
        {
            contact_resolver.resolve(&mut objects, &mut contacts, dt);
        } else
        {
            contacts.clear();
        }

        if let Some((id, point)) = held_object
        {
            let object = &objects[id];
            size.draw_line(
                &mut drawing,
                mouse_position,
                uncvt(object.transform.unproject(point)),
                0.02,
                Color{r: 50, g: 50, b: 80, a: 255}
            );
        }

        let tooltip_x = size.x as i32 / 100;
        let tooltip_y = size.y as i32 / 100;
        let tooltip_height = 25;
        let font_size = 20;
        let tooltip_color = Color{r: 255, g: 255, b: 255, a: 255};
        drawing.draw_text(
            &format!("press F to change shape: {current_shape:?}"),
            tooltip_x,
            tooltip_y,
            font_size,
            tooltip_color
        );

        drawing.draw_text(
            "press E to grab",
            tooltip_x,
            tooltip_y + tooltip_height,
            font_size,
            tooltip_color
        );

        frame_counter += 1;
        if frame_counter > (long_wait_time / dt) as u32
        {
            frame_counter = 0;
            frame_type = match frame_type
            {
                FrameType::Integration => FrameType::Collision,
                FrameType::Collision => FrameType::Integration
            };
        }

        unsafe{ raylib::ffi::WaitTime(dt as f64); }
    }
}
