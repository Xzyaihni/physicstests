use std::{
    cmp::Ordering,
    ops::ControlFlow,
    num::FpCategory
};

use raylib::prelude::*;

use nalgebra::{Matrix2, Matrix3, MatrixView2x1, Vector2 as NVector2, Vector3 as NVector3};


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

pub fn skew_symmetric(v: NVector3<f32>) -> Matrix3<f32>
{
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0
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
const ANGULAR_LIMIT: f32 = 2.0;
const VELOCITY_LOW: f32 = 0.02;
const PENETRATION_EPSILON: f32 = 0.02;
const VELOCITY_EPSILON: f32 = VELOCITY_LOW;
const SLEEP_THRESHOLD: f32 = 0.3;
const MOVEMENT_BIAS: f32 = 0.8;

const SLEEP_MOVEMENT_MAX: f32 = SLEEP_THRESHOLD * 16.0;

struct Inertias
{
    angular: f32,
    linear: f32
}

impl Inertias
{
    fn added(&self) -> f32
    {
        self.angular + self.linear
    }
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
    pub inverted: bool
}

impl IteratedMoves for PenetrationMoves
{
    fn inverted(self) -> Self
    {
        Self{
            inverted: true,
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
    pub penetration: f32
}

#[derive(Debug, Clone)]
struct AnalyzedContact
{
    pub contact: Contact,
    pub to_world: Matrix2<f32>,
    pub velocity: NVector2<f32>,
    pub desired_change: f32,
    pub a_relative: NVector2<f32>,
    pub b_relative: Option<NVector2<f32>>
}

impl AnalyzedContact
{
    fn calculate_desired_change(
        &mut self,
        objects: &[Object],
        dt: f32
    )
    {
        self.desired_change = self.contact.calculate_desired_change(objects, &self.velocity, dt);
    }

    fn inertias(&self, objects: &[Object], which: WhichObject) -> Inertias
    {
        let (object, contact_relative) = self.get(objects, which);

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
        let mut velocity_amount = penetration * inertias.linear;

        let (object, contact_relative) = self.get_mut(objects, which);

        let angular_projection = contact_relative
            + self.contact.normal * (-contact_relative).dot(&self.contact.normal);

        let angular_limit = ANGULAR_LIMIT * angular_projection.magnitude();

        if angular_amount.abs() > angular_limit
        {
            let pre_limit = angular_amount;

            angular_amount = angular_amount.clamp(-angular_limit, angular_limit);

            velocity_amount += pre_limit - angular_amount;
        }

        let velocity_change = velocity_amount * self.contact.normal;
        object.transform.position += velocity_change;

        let angular_change = if inertias.angular.classify() != FpCategory::Zero
        {
            let impulse_torque = cross_2d(
                contact_relative,
                self.contact.normal
            ) / object.inertia();

            let angular_change = impulse_torque * (angular_amount / inertias.angular);
            object.transform.rotation += angular_change;

            angular_change
        } else
        {
            0.0
        };

        PenetrationMoves{
            velocity_change,
            angular_change,
            inverted: false
        }
    }

    fn resolve_penetration(
        &self,
        objects: &mut [Object]
    ) -> (PenetrationMoves, Option<PenetrationMoves>)
    {
        let inertias = self.inertias(objects, WhichObject::A);
        let mut total_inertia = inertias.added();

        let b_inertias = self.contact.b.map(|_|
        {
            self.inertias(objects, WhichObject::B)
        });

        if let Some(ref b_inertias) = b_inertias
        {
            total_inertia += b_inertias.added();
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
    ) -> Matrix3<f32>
    {
        let (object, contact_relative) = self.get(objects, which);

        // remove this in 3d
        // !!!!!!!!!!!!!!!!!!!!!!!
        let contact_relative_3d = NVector3::new(contact_relative.x, contact_relative.y, 0.0);

        let impulse_to_torque = skew_symmetric(contact_relative_3d);
        -((impulse_to_torque / object.inertia()) * impulse_to_torque)
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

        // remove this when im actually gonna have 3d :)
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        let to_world_3d = Matrix3::new(
            self.to_world.m11, self.to_world.m12, 0.0,
            self.to_world.m21, self.to_world.m22, 0.0,
            0.0, 0.0, 1.0
        );

        let mut velocity_change = (to_world_3d.transpose() * velocity_change) * to_world_3d;

        let mut total_inverse_mass = objects[self.contact.a].physical.inverse_mass;
        if let Some(b) = self.contact.b
        {
            total_inverse_mass += objects[b].physical.inverse_mass;
        }

        let dims = 2;
        (0..dims).for_each(|i|
        {
            *velocity_change.index_mut((i, i)) += total_inverse_mass;
        });

        let impulse_local_matrix = velocity_change.try_inverse().unwrap();

        // change this in 3d
        // !!!!!!!!!!!!!!!!!!!!!!!
        let desired_change = NVector3::new(
            self.desired_change,
            -self.velocity.y,
            0.0
        );

        let mut impulse_local = impulse_local_matrix * desired_change;

        let plane_magnitude = (1..dims)
            .map(|i| impulse_local.index(i)).map(|x| x.powi(2))
            .sum::<f32>()
            .sqrt();

        let static_friction = self.contact.static_friction(objects);
        if plane_magnitude > impulse_local.x * static_friction
        {
            let friction = self.contact.dynamic_friction(objects);

            (1..dims).for_each(|i|
            {
                *impulse_local.index_mut(i) /= plane_magnitude;
            });

            // remove friction in other axes
            impulse_local.x = self.desired_change / (velocity_change.m11
                + velocity_change.m12 * friction * impulse_local.y
                + velocity_change.m13 * friction * impulse_local.z);

            (1..dims).for_each(|i|
            {
                *impulse_local.index_mut(i) *= friction * impulse_local.x;
            });
        }

        let impulse = self.to_world * impulse_local.xy();

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
        let relative_velocity = Self::velocity_from_angular(
            object.physical.angular_velocity,
            contact_relative
        ) + object.physical.velocity;

        to_world.transpose() * relative_velocity
    }

    fn restitution(&self, objects: &[Object]) -> f32
    {
        let a_restitution = objects[self.a].physical.restitution;

        self.b.map(|b|
        {
            (objects[b].physical.restitution + a_restitution) / 2.0
        }).unwrap_or(a_restitution)
    }

    fn average_physical(
        &self,
        objects: &[Object],
        f: impl Fn(&Physical) -> f32
    ) -> f32
    {
        let mut a = f(&objects[self.a].physical);
        if let Some(b) = self.b
        {
            a = (a + f(&objects[b].physical)) / 2.0;
        }

        a
    }

    // this is not how friction works irl but i dont care
    fn dynamic_friction(&self, objects: &[Object]) -> f32
    {
        self.average_physical(objects, |x| x.dynamic_friction)
    }

    fn static_friction(&self, objects: &[Object]) -> f32
    {
        self.average_physical(objects, |x| x.static_friction)
    }

    fn calculate_desired_change(
        &self,
        objects: &[Object],
        velocity_local: &NVector2<f32>,
        dt: f32
    ) -> f32
    {
        let mut acceleration_velocity = (objects[self.a].physical.last_acceleration * dt)
            .dot(&self.normal);

        if let Some(b) = self.b
        {
            acceleration_velocity -= (objects[b].physical.last_acceleration * dt).dot(&self.normal);
        }

        let restitution = if velocity_local.x.abs() < VELOCITY_LOW
        {
            0.0
        } else
        {
            self.restitution(objects)
        };

        -velocity_local.x - restitution * (velocity_local.x - acceleration_velocity)
    }

    fn awaken(&self, objects: &mut [Object])
    {
        if let Some(b) = self.b
        {
            if objects[self.a].physical.sleeping != objects[b].physical.sleeping
            {
                objects[self.a].physical.set_sleeping(false);
                objects[b].physical.set_sleeping(false);
            }
        }
    }

    fn analyze(self, objects: &[Object], dt: f32) -> AnalyzedContact
    {
        let to_world = self.to_world_matrix();

        let a_relative = self.point - objects[self.a].transform.position;
        let b_relative = self.b.map(|b| self.point - objects[b].transform.position);

        let mut velocity = Self::velocity_closing(&objects[self.a], &to_world, &a_relative);
        if let Some(b) = self.b
        {
            let b_velocity = Self::velocity_closing(
                &objects[b],
                &to_world,
                b_relative.as_ref().unwrap()
            );

            velocity -= b_velocity;
        }

        let desired_change = self.calculate_desired_change(objects, &velocity, dt);

        AnalyzedContact{
            to_world,
            velocity,
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
    fn update_iterated<Moves: IteratedMoves + Copy>(
        objects: &[Object],
        contacts: &mut [AnalyzedContact],
        moves: (Moves, Option<Moves>),
        bodies: (usize, Option<usize>),
        mut handle: impl FnMut(&[Object], &mut AnalyzedContact, Moves, NVector2<f32>)
    )
    {
        let (a_move, b_move) = moves;
        let (a_id, b_id) = bodies;

        contacts.iter_mut().for_each(|x|
        {
            let point = x.contact.point;
            let relative = |id: usize|
            {
                point - objects[id].transform.position
            };

            let this_contact_a = x.contact.a;
            let this_contact_b = x.contact.b;

            let mut handle = |move_info, contact_relative|
            {
                handle(objects, x, move_info, contact_relative);
            };

            if this_contact_a == a_id
            {
                handle(a_move, relative(this_contact_a));
            }

            if Some(this_contact_a) == b_id
            {
                handle(b_move.unwrap(), relative(this_contact_a));
            }

            if this_contact_b == Some(a_id)
            {
                handle(a_move.inverted(), relative(this_contact_b.unwrap()));
            }

            if this_contact_b.is_some() && this_contact_b == b_id
            {
                handle(b_move.unwrap().inverted(), relative(this_contact_b.unwrap()));
            }
        });
    }

    fn resolve_iterative<Moves: IteratedMoves + Copy>(
        objects: &mut [Object],
        contacts: &mut [AnalyzedContact],
        iterations: usize,
        epsilon: f32,
        compare: impl Fn(&AnalyzedContact) -> f32,
        mut resolver: impl FnMut(&mut [Object], &mut AnalyzedContact) -> (Moves, Option<Moves>),
        mut updater: impl FnMut(&[Object], &mut AnalyzedContact, Moves, NVector2<f32>)
    )
    {
        for _ in 0..iterations
        {
            if let Some((change, contact)) = contacts.iter_mut().map(|contact|
            {
                (compare(contact), contact)
            }).max_by(|(a, _), (b, _)|
            {
                a.partial_cmp(b).unwrap_or(Ordering::Less)
            }).filter(|(change, _contact)|
            {
                *change > 0.0
            })
            {
                if change > epsilon
                {
                    contact.contact.awaken(objects);
                }

                let moves = resolver(objects, contact);
                let bodies = (contact.contact.a, contact.contact.b);

                debug_assert!(moves.1.is_some() == contact.contact.b.is_some());

                Self::update_iterated::<Moves>(
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
        objects: &mut [Object],
        contacts: &mut Vec<Contact>,
        dt: f32
    )
    {
        let mut analyzed_contacts: Vec<_> = contacts.iter().cloned().map(|contact|
        {
            contact.analyze(objects, dt)
        }).collect();

        let iterations = analyzed_contacts.len() * 2;
        Self::resolve_iterative(
            objects,
            &mut analyzed_contacts,
            iterations,
            PENETRATION_EPSILON,
            |contact| contact.contact.penetration,
            |objects, contact| contact.resolve_penetration(objects),
            |_obejcts, contact, move_info, contact_relative|
            {
                let contact_change = Contact::velocity_from_angular(
                    move_info.angular_change,
                    &contact_relative
                ) + move_info.velocity_change;

                let change = contact_change.dot(&contact.contact.normal);

                if move_info.inverted
                {
                    contact.contact.penetration += change;
                } else
                {
                    contact.contact.penetration -= change;
                }
            }
        );

        Self::resolve_iterative(
            objects,
            &mut analyzed_contacts,
            iterations,
            VELOCITY_EPSILON,
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
                    contact.velocity -= change;
                } else
                {
                    contact.velocity += change;
                }

                contact.calculate_desired_change(objects, dt);
            }
        );

        contacts.clear();
    }
}

pub struct PhysicalProperties
{
    pub inverse_mass: f32,
    pub restitution: f32,
    pub damping: f32,
    pub angular_damping: f32,
    pub static_friction: f32,
    pub dynamic_friction: f32,
    pub can_sleep: bool
}

pub struct Physical
{
    inverse_mass: f32,
    restitution: f32,
    static_friction: f32,
    dynamic_friction: f32,
    can_sleep: bool,
    sleeping: bool,
    sleep_movement: f32,
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
            restitution: props.restitution,
            static_friction: props.static_friction,
            dynamic_friction: props.dynamic_friction,
            can_sleep: props.can_sleep,
            sleeping: false,
            sleep_movement: SLEEP_MOVEMENT_MAX,
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
        if self.sleeping
        {
            return;
        }

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

        if self.can_sleep
        {
            self.update_sleep_movement(dt);
        }
    }

    pub fn update_sleep_movement(&mut self, dt: f32)
    {
        let new_movement = (self.velocity.map(|x| x.powi(2)).sum() + self.angular_velocity).abs();

        let bias = MOVEMENT_BIAS.powf(dt);
        self.sleep_movement = bias * self.sleep_movement + (1.0 - bias) * new_movement;

        self.sleep_movement = self.sleep_movement.min(SLEEP_MOVEMENT_MAX);

        if self.sleep_movement < SLEEP_THRESHOLD
        {
            self.set_sleeping(true);
        }
    }

    pub fn set_sleeping(&mut self, state: bool)
    {
        if self.sleeping == state
        {
            return;
        }

        self.sleeping = state;
        if state
        {
            self.velocity = NVector2::zeros();
            self.angular_velocity = 0.0;
        } else
        {
            self.sleep_movement = SLEEP_THRESHOLD * 2.0;
        }
    }

    pub fn set_acceleration(&mut self, acceleration: NVector2<f32>)
    {
        self.acceleration = acceleration;
    }

    pub fn add_force(&mut self, force: NVector2<f32>)
    {
        self.force += force;

        self.set_sleeping(false);
    }

    fn add_force_at_point(&mut self, force: NVector2<f32>, point: NVector2<f32>)
    {
        self.add_force(force);

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

    pub fn penetration_axis(&self, other: &Self, axis: &NVector2<f32>) -> f32
    {
        let this_projected = self.rectangle_on_axis(axis);
        let other_projected = other.rectangle_on_axis(axis);

        let diff = other.transform.position - self.transform.position;

        let axis_distance = diff.dot(axis).abs();

        this_projected + other_projected - axis_distance
    }

    pub fn rectangle_rectangle_contacts<'a>(
        &'a self,
        other: &'a Self,
        this_id: usize,
        other_id: usize
    ) -> Option<Contact>
    {
        // in 3d also have to find contacts between the edges
        let dims = 2;

        let handle_penetration = move |
            this: &'a Self,
            other: &'a Self,
            a,
            b,
            axis: NVector2<f32>,
            penetration: f32
        |
        {
            move ||
            {
                let diff = other.transform.position - this.transform.position;

                let normal = if axis.dot(&diff) > 0.0
                {
                    -axis
                } else
                {
                    axis
                };

                let mut local_point = other.transform.scale / 2.0;

                (0..dims).for_each(|i|
                {
                    if other.rotation_matrix.column(i).dot(&normal) < 0.0
                    {
                        let value = -local_point.index(i);
                        *local_point.index_mut(i) = value;
                    }
                });

                let point = other.rotation_matrix * local_point + other.transform.position;

                Contact{
                    a,
                    b: Some(b),
                    point,
                    penetration,
                    normal
                }
            }
        };

        // good NAME
        let try_penetrate = |axis: MatrixView2x1<f32>| -> _
        {
            let axis: NVector2<f32> = axis.into();
            let penetration = self.penetration_axis(other, &axis);

            move |this: &'a Self, other: &'a Self, a, b| -> (f32, _)
            {
                (penetration, handle_penetration(this, other, a, b, axis, penetration))
            }
        };

        let mut penetrations = (0..dims).map(|i|
        {
            try_penetrate(self.rotation_matrix.column(i))(self, other, this_id, other_id)
        }).chain((0..dims).map(|i|
        {
            try_penetrate(other.rotation_matrix.column(i))(other, self, other_id, this_id)
        }));

        let first = penetrations.next()?;
        let least_penetrating = penetrations.try_fold(first, |b, a|
        {
            let next = if a.0 < b.0
            {
                a
            } else
            {
                b
            };

            if next.0 <= 0.0
            {
                ControlFlow::Break(())
            } else
            {
                ControlFlow::Continue(next)
            }
        });

        let (_penetration, handler) = if let ControlFlow::Continue(x) = least_penetrating
        {
            x
        } else
        {
            return None;
        };

        Some(handler())
    }
}

pub struct ZoomState
{
    zoom: f32,
    position: NVector2<f32>
}

impl ZoomState
{
    fn set_position(&mut self, position: NVector2<f32>)
    {
        self.position = position;
    }

    fn position(&self, position: NVector2<f32>) -> NVector2<f32>
    {
        let move_window = 1.0 - self.zoom;
        (position - self.position * move_window) / self.zoom
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
        self.physical.acceleration = GRAVITY;

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
            point: self.transform.position + normal * this_radius,
            penetration: this_radius + other_radius - distance,
            normal: -normal
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
            normal
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

        contacts.extend(this.rectangle_rectangle_contacts(&other, this_id, other_id));
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
                    normal
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
                        normal
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

    pub fn draw(
        &self,
        zoom: &ZoomState,
        size: WindowSize<f32>,
        drawing: &mut RaylibDrawHandle
    )
    {
        let color = if self.physical.sleeping
        {
            Color{r: 50, g: 50, b: 200, a: 255}
        } else
        {
            Color{r: 100, g: 100, b: 170, a: 255}
        };

        match self.shape
        {
            Shape::Rectangle =>
            {
                size.draw_rectangle(
                    drawing,
                    rectangle_from(
                        zoom.position(self.transform.position),
                        self.transform.scale / zoom.zoom
                    ),
                    self.transform.rotation,
                    color
                );
            },
            Shape::Circle =>
            {
                size.draw_circle(
                    drawing,
                    uncvt(zoom.position(self.transform.position)),
                    self.transform.scale.max() / 2.0 / zoom.zoom,
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

    let mut frame_counter = 0;

    let mut zoom = ZoomState{
        zoom: 1.0,
        position: NVector2::zeros()
    };

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
                restitution: 0.3,
                static_friction: 0.5,
                dynamic_friction: 0.4,
                damping: 0.9,
                angular_damping: 0.9,
                can_sleep: true
            };

            objects.push(Object::new(transform, current_shape, physical));

            holding_left = false;
            mouse_object_size = 0.0;
        }

        let long_wait = drawing.is_key_down(KeyboardKey::KEY_S);

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

        if !long_wait || frame_counter == 0
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
                            rectangle_from(rpos, scale / zoom.zoom),
                            color
                        );
                    },
                    Shape::Circle =>
                    {
                        size.draw_circle(
                            &mut drawing,
                            mouse_position,
                            mouse_object_size / zoom.zoom,
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

        let wheel = drawing.get_mouse_wheel_move();
        if wheel != 0.0
        {
            zoom.zoom = (zoom.zoom - wheel * dt).clamp(0.01, 1.0);
        }

        zoom.set_position(cvt(mouse_position));

        for id in 0..objects_len
        {
            objects[id].draw(&zoom, size, &mut drawing);
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
        }

        let mut pairs_fn = |a_id, b_id|
        {
            let (a, b) = get_two_mut(&mut objects, a_id, b_id);
            a.collide(b, &mut contacts, a_id, b_id);
        };

        {
            let mut colliders = 0..objects_len;

            // calls the function for each unique combination (excluding (self, self) pairs)
            colliders.clone().for_each(|a|
            {
                colliders.by_ref().next();
                colliders.clone().for_each(|b| pairs_fn(a, b));
            });
        }

        contacts.iter().for_each(|contact|
        {
            let start = zoom.position(contact.point);

            size.draw_circle(
                &mut drawing,
                uncvt(start),
                contact.penetration / zoom.zoom,
                Color{r: 255, g: 50, b: 50, a: 255}
            );

            size.draw_line(
                &mut drawing,
                uncvt(start),
                uncvt(start + (contact.normal / size.x * 15.0) / zoom.zoom),
                0.01,
                Color{r: 255, g: 100, b: 100, a: 255}
            );
        });

        if !long_wait || frame_counter == 0
        {
            ContactResolver::resolve(&mut objects, &mut contacts, dt);
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
        }

        unsafe{ raylib::ffi::WaitTime(dt as f64); }
    }
}
